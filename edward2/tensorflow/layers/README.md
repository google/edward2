# Bayesian Layers

Bayesian Layers is a module designed for fast experimentation with neural
network uncertainty. It extends neural network libraries with drop-in
replacements for common layers. This enables composition via a unified
abstraction over deterministic and stochastic functions and allows for
scalability via the underlying system.

For examples using Bayesian Layers, see
[`baselines/`](https://github.com/google/edward2/blob/master/baselines),
[`examples/`](https://github.com/google/edward2/blob/master/examples), and the
active research projects in
[`experimental/`](https://github.com/google/edward2/blob/master/experimental).

## 0. Motivation

In principle, the rise of AI accelerators such as TPUs lets us fit probabilistic
models at many orders of magnitude larger than state of the art. Unfortunately,
while research with uncertainty models are not limited by hardware, they are
limited by software. There are existing software supporting uncertainty models
to a limited extent. However, they remain inflexible for many practical use
cases in research. In practice, researchers often use the lower numerical
level—without a unified design for uncertainty models as there are for
deterministic neural networks. This forces researchers to reimplement even basic
methods such as Bayes by Backprop ([Blundell et al.,
2015](https://arxiv.org/abs/1505.05424))—let alone build on and scale up more
complex baselines.

## 1. Bayesian Neural Network Layers

Bayesian neural networks are neural networks with prior distributions on their
weights and biases. Like deterministic neural networks, Bayesian Layers
implements them as a composition of individual Keras layers. There are several
design points:

* __Computing the integral.__ We need to compute often-intractable integrals over weights and biases. For example, consider the variational objective for training and the approximate predictive distribution for testing:

  <img src="https://drive.google.com/uc?export=view&id=1LbURi5gIRFr6dJJFkZ2y01vAOb6VACgT" alt="integral" width="750"/>

  To enable different methods to estimate these integrals, each estimator is its own Layer. For example, the Bayesian extension of Keras' Conv2D layer has several estimators such as `ed.layers.Conv2DReparameterization` and `ed.layers.Conv2DFlipout`. Gradients for each layer estimator work automatically with `tf.GradientTape`.
* __Type Signature.__ The Bayesian extension of a deterministic layer maintains its typical
constructor arguments. It also maintains its signature for input and output Tensor shapes. This means you can swap any deterministic layer in your network with the equivalent Bayesian one, and the model type-checks (of course, more effort is required to get the new model to work).
* __Distribution over parameters.__ To specify distributions over parameters, use Keras' `kernel_initializer` and `bias_initializer`. See [`ed.initializers`](https://github.com/google/edward2/blob/master/edward2/tensorflow/initializers.py) for Bayesian Layers' built-in additions.
* __Distribution regularizers.__ To specify regularizers such as the KL penalty in variational inference, use Keras' `kernel_regularizer` and `bias_regularizer`. See [`ed.regularizers`](https://github.com/google/edward2/blob/master/edward2/tensorflow/regularizers.py) for Bayesian Layers' built-in additions.

Here's a snippet of what typical code looks like. We use a Bayesian CNN using
[TF 2.0's tutorial
architecture](https://www.tensorflow.org/tutorials/images/cnn) and trained with
variational inference.

```python
# Load and preprocess a dataset.
features, labels = ...
total_dataset_size = ...

# Define the model.
model = tf.python.keras.Sequential([
  ed.layers.Conv2DFlipout(32, (3, 3), activation='relu'),
  tf.python.keras.layers.MaxPooling2D((2, 2)),
  ed.layers.Conv2DFlipout(64, (3, 3), activation='relu'),
  tf.python.keras.layers.MaxPooling2D((2, 2)),
  ed.layers.Conv2DFlipout(64, (3, 3), activation='relu'),
  tf.python.keras.layers.Flatten(),
  ed.layers.DenseVariationalDropout(64, activation='relu'),
  ed.layers.DenseVariationalDropout(10),
])

# Specify custom loss function and run training loop. Or use model.compile and
# model.fit, scaling the losses term by total_dataset_size.
def loss_fn(features, labels):
  logits = model(features)
  nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
  kl = sum(model.losses) / total_dataset_size
  return nll + kl

num_steps = 1000
for _ in range(num_steps):
  with tf.GradientTape() as tape:
    loss = loss_fn(features, labels)
  gradients = tape.gradient(loss, model.variables)  # use any optimizer here
```

For testing, there are two popular approaches: Monte Carlo averaging and
heuristics that require only a single forward pass. (Other approaches are also
supported.)

```python
test_features = ...

# Compute the averaged prediction across multiple samples.
num_samples = 10
logits = tf.reduce_mean([model(test_features) for _ in range(num_samples)],
                        axis=0)
predicted_labels = tf.argmax(logits, axis=-1)

# Use only one forward pass at test time, setting each of the trained weights
# to their distribution's mean.
def take_mean(f, *args, **kwargs):
  """Tracer which sets each random variable's value to its mean."""
  rv = f(*args, **kwargs)
  rv._value = rv.distribution.mean()
  return rv

with ed.trace(take_mean):
  logits = model(test_features)
predicted_labels = tf.argmax(logits, axis=-1)
```

## 2. Gaussian Process Layers

As opposed to representing distributions over functions through the weights,
Gaussian processes represent distributions over functions by specifying the
value of the function at different inputs. GPs have the same design points:

* __Computing the integral.__ Each estimator is its own Layer. This includes `ed.layers.GaussianProcess` for exact (albeit expensive) integration and `ed.layers.SparseGaussianProcess` for inducing variable approximations.
* __Type Signature.__ For the equivalent deterministic layer, GPs maintain its typical arguments as well as tensor-shaped inputs and outputs. For example, `units` in a Gaussian process layer determine the GP's output dimensionality, where `ed.layers.GaussianProcess(32)` is the Bayesian nonparametric extension of `tf.python.keras.layers.Dense(32)`. Instead of an `activation` function argument, GP layers have mean and covariance function arguments which default to the zero function and squared exponential kernel respectively.
* __Distribution regularizers.__ To specify regularizers such as the KL penalty in variational inference, use Keras' `kernel_regularizer` and `bias_regularizer`. See [`ed.regularizers`](https://github.com/google/edward2/blob/master/edward2/tensorflow/regularizers.py) for Bayesian Layers' built-in additions.

Here's a snippet of what typical code looks like. We use a 3-layer deep GP
trained with variational inference.

```python
# Define the model.
model = tf.python.keras.Sequential([
  tf.python.keras.layers.Flatten(),
  ed.layers.SparseGaussianProcess(256, num_inducing=512),
  ed.layers.SparseGaussianProcess(256, num_inducing=512),
  ed.layers.SparseGaussianProcess(3, num_inducing=512),
])
predictions = model(features)

# Specify custom loss function and run training loop. Or use model.compile and
# model.fit, scaling the losses term by total_dataset_size.
def loss_fn(features, labels):
  logits = model(features)
  nll = -tf.reduce_mean(predictions.distribution.log_prob(labels))
  kl = sum(model.losses) / total_dataset_size
  return nll + kl
```

For training, use typical setups for TensorFlow 2.0. For testing, use Monte
Carlo averages or a single forward pass approximation as shown above for
Bayesian neural networks. Note using the exact Gaussian process layer as a
model by itself will not require these approximations.

## 3. Stochastic Output Layers

In addition to uncertainty over the _mapping_ defined by a layer, we may want to
simply add stochasticity to the output. These outputs have a tractable
distribution, and we often would like to access its properties: for example,
auto-encoding with stochastic encoders and decoders; or a dynamics model whose
network output is a discretized mixture density.

Given a Tensor input, stochastic output layers perform deterministic
computations and return an `ed.RandomVariable`.  Stochastic output layers
typically don't have mandatory constructor arguments. An optional `units`
argument determines its output dimensionality (operated on via a trainable
linear projection); the default maintains the input shape and has no such
projection.

Here's a snippet of what typical code looks like. We use a variational-
autoencoder.

```python
# Define the model.
encoder = tf.python.keras.Sequential([
  tf.python.keras.layers.Conv2D(128, 5, 1, padding='same', activation='relu'),
  tf.python.keras.layers.Conv2D(128, 5, 2, padding='same', activation='relu'),
  tf.python.keras.layers.Conv2D(512, 7, 1, padding='valid', activation='relu'),
  ed.layers.Normal(name='latent_code'),
])
decoder = tf.python.keras.Sequential([
  tf.python.keras.layers.Conv2DTranspose(256, 7, 1, padding='valid', activation='relu'),
  tf.python.keras.layers.Conv2DTranspose(128, 5, 2, padding='same', activation='relu'),
  tf.python.keras.layers.Conv2DTranspose(128, 5, 1, padding='same', activation='relu'),
  tf.python.keras.layers.Conv2D(3*256, 5, 1, padding='same', activation=None),
  tf.python.keras.layers.Reshape([256, 256, 3, 256]),
  ed.layers.Categorical(name='image'),
])

# Specify custom loss function and run training loop. Or use model.compile and
# model.fit.
def loss_fn(features):
  encoding = encoder(features)
  nll = -decoder(encoding).log_prob(features)
  kl = encoding.distribution.kl_divergence(ed.Normal(0., 1.).distribution)
  return tf.reduce_mean(nll + kl)
```

For training and testing, use typical setups for TensorFlow 2.0. Note testing
does not require Monte Carlo averaging like BNN and GP layers (unless you
include those layers in your model).

## 4. Reversible Layers

With random variables in layers, one can naturally capture invertible neural
networks which propagate uncertainty from input to output. This allows one to
perform transformations of random variables, ranging from simple transformations
such as for a log-normal distribution or high-dimensional transformations for
flow-based models. There are two design points:

* __Inversion.__ To enable invertible neural networks, we overload the notion of a layer by adding an additional method `reverse` which performs the inverse computation of its call and optionally `log_det_jacobian`. Higher-order layers also exist. For example, `ed.layers.Reverse` takes a layer as input and returns another layer swapping the forward and reverse computation.
* __Propagating Uncertainty.__ As with other deterministic layers, reversible layers are Tensor-input Tensor-output. In order to propagate uncertainty from input to output, reversible layers may also take a `RandomVariable` as input and return a transformed `RandomVariable` determined by its call, `reverse`, and `log_det_jacobian`.

Here's a snippet of what typical code looks like. We use a discrete flow
over 64-dimensional sequences.

```python
sequence_length, vocab_size = ...

# Define the model.
flow = tf.python.keras.Sequential([
  ed.layers.DiscreteAutoregressiveFlow(ed.layers.MADE(vocab_size, hidden_dims=[256, 256])),
  ed.layers.DiscreteAutoregressiveFlow(ed.layers.MADE(vocab_size, hidden_dims=[256, 256], order='right-to-left')),
  ed.layers.DiscreteAutoregressiveFlow(ed.layers.MADE(vocab_size, hidden_dims=[256, 256])),
])
base = ed.Categorical(logits=tf.Variable(tf.random.normal([sequence_length, vocab_size]))

# Specify custom loss function and run training loop. Or use model.compile and
# model.fit.
def loss_fn(features):
  whitened_features = flow.reverse(features)
  # In this example, we don't include log-det-jacobian as in continuous flows.
  # Discrete flows don't require them.
  return -tf.reduce_mean(base.distribution.log_prob(whitened_features))
```

For training and testing, use typical setups for TensorFlow 2.0. Note testing
does not require Monte Carlo averaging like BNN and GP layers (unless you
also include those layers in your model).

## 5. Other Layers

<!--
TODO(trandustin): Add explicit link to http://edwardlib.org/api after
edwardlib.org is updated.
-->

See the API documentation for a comprehensive list of layers. These include
noise contrastive prior layers like `ed.layers.NCPNormalPerturb` for additive
Gaussian noise, and normalization layers like `ed.layers.ActNorm` which is
helpful for normalizing flows.

## References

> Tran, D., Dusenberry M. W., van der Wilk M., Hafner D. (2019).
> [Bayesian Layers: A Module for Neural Network Uncertainty](https://arxiv.org/abs/1812.03973).
> In _Neural Information Processing Systems_.

```none
@inproceedings{tran2019bayesian,
  author = {Dustin Tran and Michael W. Dusenberry and Danijar Hafner and Mark van der Wilk},
  title={Bayesian {L}ayers: A module for neural network uncertainty},
  booktitle = {Neural Information Processing Systems},
  year={2019}
}
```
