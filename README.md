# Edward2

Edward2 is a _simple_ probabilistic programming language. It provides core
utilities in deep learning ecosystems so that one can write models as
probabilistic programs and manipulate a model's computation for flexible
training and inference. It's organized as follows:

* [`edward2/`](https://github.com/google/edward2/blob/main/edward2/):
  Library code.
* [`examples/`](https://github.com/google/edward2/blob/main/examples):
  Examples.
* [`experimental/`](https://github.com/google/edward2/blob/main/experimental):
  Active research projects.

Are you upgrading from Edward? Check out the guide
[`Upgrading_from_Edward_to_Edward2.md`](https://github.com/google/edward2/blob/main/Upgrading_From_Edward_To_Edward2.md).
The core utilities are fairly low-level: if you'd like a high-level module for
uncertainty modeling, check out the guide for
[Bayesian Layers](https://github.com/google/edward2/tree/main/edward2/tensorflow/layers).
We recommend the
[Uncertainty Baselines](https://github.com/google/uncertainty-baselines)
if you'd like to build on research-ready code.

## Installation

To install the latest stable version, run

```sh
pip install edward2
```

To install the latest development version, run

```sh
pip install "git+https://github.com/google/edward2.git#egg=edward2"
```

Edward2 supports three backends: TensorFlow (the default), JAX, and NumPy ([see
below to activate](#using-the-jax-or-numpy-backend)). Installing `edward2` does
not automatically install any backend. To get these dependencies, use for
example `pip install edward2[tensorflow]"`, replacing `tensorflow` for the
appropriate backend. Sometimes Edward2 uses the latest changes from TensorFlow
in which you'll need TensorFlow's nightly package: use `pip install edward2[tf-
nightly]`.

## 1. Models as Probabilistic Programs

### Random Variables

In Edward2, we use
[`RandomVariables`](https://github.com/google/edward2/blob/main/edward2/tensorflow/random_variable.py)
to specify a probabilistic model's structure.
A random variable `rv` carries a probability distribution (`rv.distribution`),
which is a TensorFlow Distribution instance governing the random variable's methods
such as `log_prob` and `sample`.

Random variables are formed like TensorFlow Distributions.

```python
import edward2 as ed

normal_rv = ed.Normal(loc=0., scale=1.)
## <ed.RandomVariable 'Normal/' shape=() dtype=float32 numpy=0.0024812892>
normal_rv.distribution.log_prob(1.231)
## <tf.Tensor: id=11, shape=(), dtype=float32, numpy=-1.6766189>

dirichlet_rv = ed.Dirichlet(concentration=tf.ones([2, 3]))
## <ed.RandomVariable 'Dirichlet/' shape=(2, 3) dtype=float32 numpy=
array([[0.15864784, 0.01217205, 0.82918006],
       [0.23385087, 0.69622266, 0.06992647]], dtype=float32)>
```

By default, instantiating a random variable `rv` creates a sampling op to form
the tensor `rv.value ~ rv.distribution.sample()`. The default number of samples
(controllable via the `sample_shape` argument to `rv`) is one, and if the
optional `value` argument is provided, no sampling op is created. Random
variables can interoperate with TensorFlow ops: the TF ops operate on the sample.

```python
x = ed.Normal(loc=tf.zeros(2), scale=tf.ones(2))
y = 5.
x + y, x / y
## (<tf.Tensor: id=109, shape=(2,), dtype=float32, numpy=array([3.9076924, 4.588356 ], dtype=float32)>,
##  <tf.Tensor: id=111, shape=(2,), dtype=float32, numpy=array([-0.21846154, -0.08232877], dtype=float32)>)
tf.tanh(x * y)
## <tf.Tensor: id=114, shape=(2,), dtype=float32, numpy=array([-0.99996394, -0.9679181 ], dtype=float32)>
x[1]  # 2nd normal rv
## <ed.RandomVariable 'Normal/' shape=() dtype=float32 numpy=-0.41164386>
```

### Probabilistic Models

Probabilistic models in Edward2 are expressed as Python functions that
instantiate one or more `RandomVariables`. Typically, the function ("program")
executes the generative process and returns samples. Inputs to the
function can be thought of as values the model conditions on.

Below we write Bayesian logistic regression, where binary outcomes are generated
given features, coefficients, and an intercept. There is a prior over the
coefficients and intercept. Executing the function adds operations samples
coefficients and intercept from the prior and uses these samples to compute the
outcomes.

```python
def logistic_regression(features):
  """Bayesian logistic regression p(y | x) = int p(y | x, w, b) p(w, b) dwdb."""
  coeffs = ed.Normal(loc=tf.zeros(features.shape[1]), scale=1., name="coeffs")
  intercept = ed.Normal(loc=0., scale=1., name="intercept")
  outcomes = ed.Bernoulli(
      logits=tf.tensordot(features, coeffs, [[1], [0]]) + intercept,
      name="outcomes")
  return outcomes

num_features = 10
features = tf.random.normal([100, num_features])
outcomes = logistic_regression(features)
# <ed.RandomVariable 'outcomes/' shape=(100,) dtype=int32 numpy=
# array([1, 0, ... 0, 1], dtype=int32)>
```

Edward2 programs can also represent distributions beyond those which directly
model data. For example, below we write a learnable distribution with the
intention to approximate it to the logistic regression posterior.

```python
def logistic_regression_posterior(coeffs_loc, coeffs_scale,
                                  intercept_loc, intercept_scale):
  """Posterior of Bayesian logistic regression p(w, b | {x, y})."""
  coeffs = ed.MultivariateNormalTriL(
      loc=coeffs_loc,
      scale_tril=tfp.trainable_distributions.tril_with_diag_softplus_and_shift(
          coeffs_scale),
      name="coeffs_posterior")
  intercept = ed.Normal(
      loc=intercept_loc,
      scale=tf.nn.softplus(intercept_scale) + 1e-5,
      name="intercept_posterior")
  return coeffs, intercept

coeffs_loc = tf.Variable(tf.random.normal([num_features]))
coeffs_scale = tf.Variable(tf.random.normal(
    [num_features*(num_features+1) // 2]))

intercept_loc = tf.Variable(tf.random.normal([]))
intercept_scale = tf.Variable(tf.random.normal([]))
posterior_coeffs, posterior_intercept = logistic_regression_posterior(
    coeffs_loc, coeffs_scale, intercept_loc, intercept_scale)
```

## 2. Manipulating Model Computation

### Tracing

Training and testing probabilistic models typically require more than just
samples from the generative process. To enable flexible training and testing, we
manipulate the model's computation using
[tracing](https://github.com/google/edward2/blob/main/edward2/tracer.py).

A tracer is a function that acts on another function `f` and its arguments
`*args`, `**kwargs`. It performs various computations before returning an output
(typically `f(*args, **kwargs)`: the result of applying the function itself).
The `ed.trace` context manager pushes tracers onto a stack, and any
traceable function is intercepted by the stack. All random variable
constructors are traceable.

Below we trace the logistic regression model's generative process. In
particular, we make predictions with its learned posterior means rather than
with its priors.

```python
def set_prior_to_posterior_mean(f, *args, **kwargs):
  """Forms posterior predictions, setting each prior to its posterior mean."""
  name = kwargs.get("name")
  if name == "coeffs":
    return posterior_coeffs.distribution.mean()
  elif name == "intercept":
    return posterior_intercept.distribution.mean()
  return f(*args, **kwargs)

with ed.trace(set_prior_to_posterior_mean):
  predictions = logistic_regression(features)

training_accuracy = (
    tf.reduce_sum(tf.cast(tf.equal(predictions, outcomes), tf.float32)) /
    tf.cast(outcomes.shape[0], tf.float32))
```

### Program Transformations

Using tracing, one can also apply program transformations, which map
from one representation of a model to another. This provides convenient access
to different model properties depending on the downstream use case.

For example, Markov chain Monte Carlo algorithms often require a model's
log-joint probability function as input. Below we take the Bayesian logistic
regression program which specifies a generative process, and apply the built-in
`ed.make_log_joint` transformation to obtain its log-joint probability function.
The log-joint function takes as input the generative program's original inputs
as well as random variables in the program. It returns a scalar Tensor
summing over all random variable log-probabilities.

In our example, `features` and `outcomes` are fixed, and we want to use
Hamiltonian Monte Carlo to draw samples from the posterior distribution of
`coeffs` and `intercept`. To this use, we create `target_log_prob_fn`, which
takes just `coeffs` and `intercept` as arguments and pins the input `features`
and output rv `outcomes` to its known values.

```python
import no_u_turn_sampler  # local file import

# Set up training data.
features = tf.random.normal([100, 55])
outcomes = tf.random.uniform([100], minval=0, maxval=2, dtype=tf.int32)

# Pass target log-probability function to MCMC transition kernel.
log_joint = ed.make_log_joint_fn(logistic_regression)

def target_log_prob_fn(coeffs, intercept):
  """Target log-probability as a function of states."""
  return log_joint(features,
                   coeffs=coeffs,
                   intercept=intercept,
                   outcomes=outcomes)

coeffs_samples = []
intercept_samples = []
coeffs = tf.random.normal([55])
intercept = tf.random.normal([])
target_log_prob = None
grads_target_log_prob = None
for _ in range(1000):
  [
      [coeffs, intercepts],
      target_log_prob,
      grads_target_log_prob,
  ] = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[coeffs, intercept],
          step_size=[0.1, 0.1],
          current_target_log_prob=target_log_prob,
          current_grads_target_log_prob=grads_target_log_prob)
  coeffs_samples.append(coeffs)
  intercept_samples.append(coeffs)
```

The returned `coeffs_samples` and `intercept_samples` contain 1,000 posterior
samples for `coeffs` and `intercept` respectively. They may be used, for
example, to evaluate the model's posterior predictive on new data.

## Using the JAX or NumPy backend

Using alternative backends is as simple as the following:

```python
import edward2.numpy as ed  # NumPy backend
import edward2.jax as ed  # or, JAX backend
```

In the NumPy backend, Edward2 wraps SciPy distributions. For example, here's
linear regression.

```python
def linear_regression(features, prior_precision):
  beta = ed.norm.rvs(loc=0.,
                     scale=1. / np.sqrt(prior_precision),
                     size=features.shape[1])
  y = ed.norm.rvs(loc=np.dot(features, beta), scale=1., size=1)
  return y
```

## References

In general, we recommend citing the following article.

> Tran, D., Hoffman, M. D., Moore, D., Suter, C., Vasudevan S., Radul A.,
> Johnson M., and Saurous R. A. (2018).
> [Simple, Distributed, and Accelerated Probabilistic Programming](https://arxiv.org/abs/1811.02091).
> In _Neural Information Processing Systems_.

```none
@inproceedings{tran2018simple,
  author = {Dustin Tran and Matthew D. Hoffman and Dave Moore and Christopher Suter and Srinivas Vasudevan and Alexey Radul and Matthew Johnson and Rif A. Saurous},
  title = {Simple, Distributed, and Accelerated Probabilistic Programming},
  booktitle = {Neural Information Processing Systems},
  year = {2018},
}
```

If you'd like to cite the layers module specifically, use the following article.

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
