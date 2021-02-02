# Upgrading from Edward to Edward2

This guide outlines how to port code from the
[Edward](http://edwardlib.org/)
probabilistic programming system to
[Edward2](https://github.com/google/edward2).
We recommend Edward users use Edward2 for specifying models and other TensorFlow
primitives for performing downstream computation.

Edward2 is a distillation of Edward. It is a low-level language for specifying
probabilistic models as programs and manipulating their computation.
Probabilistic inference, criticism, and any other part of the scientific process
(Box, 1976) use arbitrary TensorFlow ops. Their associated abstractions live in
the TensorFlow ecosystem and do not strictly require Edward2.

Are you having difficulties upgrading to Edward2? Raise a
[GitHub issue](https://github.com/google-research/google-research/issues)
and we're happy to help. Alternatively, if you have tips, feel free to send a
pull request to improve this guide.

## Namespaces

__Edward__.

```python
import edward as ed
from edward.models import Empirical, Gamma, Poisson

dir(ed)
## ['criticisms',
##  'inferences',
##  'models',
##  'util',
##   ...,  # criticisms in global namespace for convenience
##   ...,  # inference algorithms in global namespace for convenience
##   ...]  # utility functions in global namespace for convenience
```

__Edward2__.

```python
import edward2 as ed

dir(ed)
## [...,  # random variables
##  'layers',  # Bayesian Layers (Tran et al., 2019)
##  'initializers',
##  'regularizers',
##  'constraints',
##  'condition',  # tools for manipulating program execution
##  'get_next_tracer',
##  'make_log_joint_fn',
##  'make_random_variable',
##  'tape',
##  'trace',
##  'traceable']
```

## Probabilistic Models

__Edward__. You write models inline with any other code, composing
random variables. As illustration, consider a deep exponential family
(Ranganath et al., 2015).

```python
bag_of_words = np.random.poisson(5., size=[256, 32000])  # training data as matrix of counts
data_size, feature_size = bag_of_words.shape  # number of documents x words (vocabulary)
units = [100, 30, 15]  # number of stochastic units per layer
shape = 0.1  # Gamma shape parameter

w2 = Gamma(0.1, 0.3, sample_shape=[units[2], units[1]])
w1 = Gamma(0.1, 0.3, sample_shape=[units[1], units[0]])
w0 = Gamma(0.1, 0.3, sample_shape=[units[0], feature_size])

z2 = Gamma(0.1, 0.1, sample_shape=[data_size, units[2]])
z1 = Gamma(shape, shape / tf.matmul(z2, w2))
z0 = Gamma(shape, shape / tf.matmul(z1, w1))
x = Poisson(tf.matmul(z1, w0))
```

__Edward2__. You write models as functions, where
random variables operate with the same behavior as Edward's.

```python
def deep_exponential_family(data_size, feature_size, units, shape):
  """A multi-layered topic model over a documents-by-terms matrix."""
  w2 = ed.Gamma(0.1, 0.3, sample_shape=[units[2], units[1]], name="w2")
  w1 = ed.Gamma(0.1, 0.3, sample_shape=[units[1], units[0]], name="w1")
  w0 = ed.Gamma(0.1, 0.3, sample_shape=[units[0], feature_size], name="w0")

  z2 = ed.Gamma(0.1, 0.1, sample_shape=[data_size, units[2]], name="z2")
  z1 = ed.Gamma(shape, shape / tf.matmul(z2, w2), name="z1")
  z0 = ed.Gamma(shape, shape / tf.matmul(z1, w1), name="z0")
  x = ed.Poisson(tf.matmul(z0, w0), name="x")
  return x
```

Broadly, the function's outputs capture what the probabilistic program is over
(the `y` in `p(y | x)`), and the function's inputs capture what the
probabilistic program conditions on (the `x` in `p(y | x)`). Note it's best
practice to write names to all random variables: this is useful for cleaner
names in the computational graph as well as for manipulating model computation.

## Graph vs Eager Execution

__Edward__. In TensorFlow graph mode, you fetch values from the TensorFlow graph
using a built-in Edward session. Eager mode is not available.

```python
# Generate from model: returns np.ndarray of shape (data_size, feature_size).
with ed.get_session() as sess:
  sess.run(x)
```

__Edward2__. Edward2 operates with TensorFlow 2.0. It always uses eager
execution so there is no TensorFlow session.

```python
# Generate from model: returns tf.Tensor of shape (data_size, feature_size).
x = deep_exponential_family(data_size, feature_size, units, shape)
x.numpy()  # converts from eagerly executed tf.Tensor to np.ndarray
```

## Probabilistic Inference

In Edward, there is a taxonomy of inference algorithms, with many built-in from
the abstract classes of `ed.MonteCarlo` (sampling) and `ed.VariationalInference`
(optimization). In Edward2, inference algorithms are modularized
so that they can depend on arbitrary TensorFlow ops; any associated abstractions
do not live in Edward2. Below we outline variational inference and Markov
chain Monte Carlo.

### Variational Inference

__Edward__. You construct random variables with free parameters, representing
the model's posterior approximation. You align these random variables together
with the model's and construct an inference class.

```python
def trainable_positive_pointmass(shape, name=None):
  """Learnable point mass distribution over positive reals."""
  with tf.variable_scope(None, default_name="trainable_positive_pointmass"):
    return PointMass(tf.nn.softplus(tf.get_variable("mean", shape)), name=name)

def trainable_gamma(shape, name=None):
  """Learnable Gamma via shape and scale parameterization."""
  with tf.variable_scope(None, default_name="trainable_gamma"):
    return Gamma(tf.nn.softplus(tf.get_variable("shape", shape)),
                 1.0 / tf.nn.softplus(tf.get_variable("scale", shape)),
                 name=name)

qw2 = trainable_positive_pointmass(w2.shape)
qw1 = trainable_positive_pointmass(w1.shape)
qw0 = trainable_positive_pointmass(w0.shape)
qz2 = trainable_gamma(z2.shape)
qz1 = trainable_gamma(z1.shape)
qz0 = trainable_gamma(z0.shape)

inference = ed.KLqp({w0: qw0, w1: qw1, w2: qw2, z0: qz0, z1: qz1, z2: qz2},
                    data={x: bag_of_words})
```

To schedule training, you call `inference.run()` which automatically
handles the schedule. Alternatively, you manually schedule training with
`inference`'s class methods.

```python
inference.initialize(n_iter=10000)

tf.global_variables_initializer().run()
for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

inference.finalize()
```

__Edward2__. You set up variational inference manually and/or build your own
abstractions.

Below we use Edward2's
[tracing](https://github.com/google/edward2/blob/master/edward2/trace.py)
in order to manipulate model computation. We define the variational
approximation—another Edward2 program—and apply tracers to write the
evidence lower bound (Hinton & Camp, 1993; Jordan, Ghahramani, Jaakkola, & Saul,
1999; Waterhouse, MacKay, & Robinson, 1996). Note we use factory functions
(functions which build other functions) for simplicity, but you can also use
`tf.python.keras.Models` as stateful classes which automatically manage the variables.

```python
def build_trainable_positive_pointmass(shape, name=None):
  """Builds point mass r.v. over positive reals and its parameters."""
  mean = tf.Variable(tf.random.normal(shape))
  def positive_pointmass():
    return ed.PointMass(tf.nn.softplus(mean), name=name)
  return positive_pointmass, [mean]

def build_trainable_gamma(shape, name=None):
  """Builds Gamma random variable and its parameters."""
  shape_param = tf.Variable(tf.random.normal(shape))
  scale_param = tf.Variable(tf.random.normal(shape))
  def gamma():
    return ed.Gamma(tf.nn.softplus(shape_param), 1./tf.nn.softplus(scale_param), name=name)
  return gamma, [shape_param, scale_param]

def build_deep_exponential_family_variational():
  """Builds posterior approximation q(w{0,1,2}, z{1,2,3} | x) and parameters."""
  QW2, qw2_params = build_trainable_positive_pointmass(w2.shape, name="qw2")
  QW1, qw1_params = build_trainable_positive_pointmass(w1.shape, name="qw1")
  QW0, qw0_params = build_trainable_positive_pointmass(w0.shape, name="qw0")
  QZ2, qz2_params = build_trainable_gamma(z2.shape, name="qz2")
  QZ1, qz1_params = build_trainable_gamma(z1.shape, name="qz1")
  QZ0, qz0_params = build_trainable_gamma(z0.shape, name="qz0")
  parameters = (qw2_params + qw1_params + qw0_params +
                qz2_params + qz1_params + qz0_params)
  def deep_exponential_family_variational():
    return QW2(), QW1(), QW0(), QZ2(), QZ1(), QZ0()
  return deep_exponential_family_variational, parameters
```

To schedule training, you use typical TensorFlow. For an equivalent
`inference.run()`-like API, see Keras'
[`model.compile` and `model.fit` high-level API](https://www.tensorflow.org/guide/keras/overview)).
Below uses a custom training loop.

```python
max_steps = 10000  # number of training iterations
model_dir = None  # directory for model checkpoints

writer = tf.summary.create_file_writer(model_dir)
[deep_exponential_family_variational,
 trainable_variables] = build_deep_exponential_family_variational()

@tf.function
def train_step(bag_of_words, step):
  with tf.GradientTape() as tape:
    # Compute expected log-likelihood. First, sample from the variational
    # distribution; second, compute the log-likelihood given the sample.
    qw2, qw1, qw0, qz2, qz1, qz0 = deep_exponential_family_variational()

    # Compute forward pass of model, setting value of the priors to the
    # approximate posterior samples. We also record the forward pass' execution
    # via ed.tape().
    with ed.tape() as model_tape:
      with ed.condition(w2=qw2, w1=qw1, w0=qw0,
                        z2=qz2, z1=qz1, z0=qz0):
        posterior_predictive = deep_exponential_family(data_size, feature_size, units, shape)

    log_likelihood = posterior_predictive.distribution.log_prob(bag_of_words)

    # Compute analytic KL-divergence between variational and prior distributions.
    kl = 0.
    for rv_name, variational_rv in [("z0", qz0), ("z1", qz1), ("z2", qz2),
                                    ("w0", qw0), ("w1", qw1), ("w2", qw2)]:
      kl += tf.reduce_sum(variational_rv.distribution.kl_divergence(
          model_tape[rv_name].distribution))

    elbo = tf.reduce_mean(log_likelihood - kl)
    with writer.default():
      tf.summary.scalar("elbo", elbo, step=step)
     loss = -elbo
  optimizer = tf.python.keras.optimizers.Adam(1e-3)
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss

for step in range(max_steps):
  start_time = time.time()
  bag_of_words = next(train_data)
  loss = train_step(bag_of_words, step)
  if step % 500 == 0:
    writer.flush()
    duration = time.time() - start_time
    print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
        step, elbo_value, duration))
```

### Markov chain Monte Carlo

__Edward__. Similar to variational inference, you construct random variables
with free parameters, representing the model's posterior approximation. You
align these random variables together with the model's and construct an
inference class.

```python
num_samples = 10000  # number of events to approximate posterior

qw2 = Empirical(tf.get_variable("qw2/params", [num_samples, units[2], units[1]]))
qw1 = Empirical(tf.get_variable("qw1/params", [num_samples, units[1], units[0]]))
qw0 = Empirical(tf.get_variable("qw0/params", [num_samples, units[0], feature_size]))
qz2 = Empirical(tf.get_variable("qz2/params", [num_samples, data_size, units[2]]))
qz1 = Empirical(tf.get_variable("qz1/params", [num_samples, data_size, units[1]]))
qz0 = Empirical(tf.get_variable("qz0/params", [num_samples, data_size, units[0]]))

inference = ed.HMC({w0: qw0, w1: qw1, w2: qw2, z0: qz0, z1: qz1, z2: qz2},
                   data={x: bag_of_words})
```

You use the inference class' methods to schedule training.

__Edward2__. Use, e.g., a transition kernel which may be a Tensor-in Tensor-out
function propagating from one state to the next. Apply that transition kernel
over multiple iterations until convergence.

Below we first rewrite the Edward2 model in terms of its target log-probability
as a function of latent variables. Namely, it is the model's log-joint
probability function with fixed hyperparameters and observations anchored at the
data. We then apply a function which executes a
Hamiltonian Monte Carlo transition kernel to return a collection of new states
given previous states.

```python
import no_u_turn_sampler  # local file import

num_samples = 10000  # number of events to approximate posterior
qw2 = tf.nn.softplus(tf.random.normal([units[2], units[1]]))  # initial state
qw1 = tf.nn.softplus(tf.random.normal([units[1], units[0]]))
qw0 = tf.nn.softplus(tf.random.normal([units[0], feature_size]))
qz2 = tf.nn.softplus(tf.random.normal([data_size, units[2]]))
qz1 = tf.nn.softplus(tf.random.normal([data_size, units[1]]))
qz0 = tf.nn.softplus(tf.random.normal([data_size, units[0]]))

log_joint = ed.make_log_joint_fn(deep_exponential_family)

def target_log_prob_fn(w2, w1, w0, z2, z1, z0):
  """Target log-probability as a function of states."""
  return log_joint(data_size, feature_size, units, shape,
                   w2=w2, w1=w1, w0=w0, z2=z2, z1=z1, z0=z0, x=bag_of_words)

target_log_prob = grads_target_log_prob = None
for _ in range(num_samples):
  [
      [qw2, qw1, qw0, qz2, qz1, qz0],
      target_log_prob,
      grads_target_log_prob,
  ] = no_u_turn_sampler.kernel(
      target_log_prob_fn=target_log_prob_fn,
      current_state=[qw2, qw1, qw0, qz2, qz1, qz0],
      step_size=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
      current_target_log_prob=target_log_prob,
      current_grads_target_log_prob=grads_target_log_prob)
```

Tracking Markov chain Monte Carlo diagnostics is the same workflow as in
variational inference. Instead of tracking a loss function, however, one uses,
for example, a counter for the number of accepted samples. This lets us monitor
a running statistic of MCMC's acceptance rate.

## Model & Inference Criticism

__Edward__. You typically use two functions: `ed.evaluate` for assessing how
model predictions match the true data; and `ed.ppc` for
assessing how data generated from the model matches the true data.

```python
# Build posterior predictive: it is parameterized by a variational posterior sample.
posterior_predictive = ed.copy(
    x, {w0: qw0, w1: qw1, w2: qw2, z0: qz0, z1: qz1, z2: qz2})

# Evaluate average log-likelihood of data.
ed.evaluate('log_likelihood', data={posterior_predictive: bag_of_words})
## np.ndarray of shape ()

# Compare TF-IDF on real vs generated data.
def tfidf(bag_of_words):
  """Computes term-frequency inverse-document-frequency."""
  num_documents = bag_of_words.shape[0]
  idf = tf.log(num_documents) - tf.log(tf.count_nonzero(bag_of_words, axis=0))
  return bag_of_words * idf

observed_statistics, replicated_statistics = ed.ppc(
    lambda data, latent_vars: tf_idx(data[posterior_predictive]),
    {posterior_predictive: bag_of_words},
    n_samples=100)
```

__Edward2__. Build the metric manually or use TensorFlow
abstractions such as `tf.python.keras.metrics`.

```python
# See posterior_predictive built in Variational Inference section.
log_likelihood = tf.reduce_mean(posterior_predictive.distribution.log_prob(bag_of_words))
## tf.Tensor of shape ()

# Compare statistics by sampling from model in a for loop.
observed_statistic = tfidf(bag_of_words)
replicated_statistics = [tfidf(posterior_predictive) for _ in range(100)]
```

## References

1. George Edward Pelham Box. Science and statistics. _Journal of the American Statistical Association_, 71(356), 791–799, 1976.
2. Hinton, G. E., & Camp, D. van. (1993). Keeping the neural networks simple by minimizing the description length of the weights. In Conference on learning theory. ACM.
3. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine Learning, 37(2), 183–233.
4. Rajesh Ranganath, Linpeng Tang, Laurent Charlin, David M. Blei. Deep exponential families. In _Artificial Intelligence and Statistics_, 2015.
5. Dustin Tran, Michael W. Dusenberry, Mark van der Wilk, Danijar Hafner. Bayesian Layers: A Module for Neural Network Uncertainty. In _Neural Information Processing Systems_, 2019.
6. Waterhouse, S., MacKay, D., & Robinson, T. (1996). Bayesian methods for mixtures of experts. Advances in Neural Information Processing Systems, 351–357.
