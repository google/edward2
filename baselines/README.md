# Uncertainty Baselines

The goal of Uncertainty Baselines is to provide a template for researchers to build on. The baselines can be a starting point for any new ideas, applications, and/or for communicating with other uncertainty researchers. This is done in four ways:

1. Provide high-quality implementations of standard and state-of-the-art methods on standard tasks.
2. Have minimal dependencies on other files in the codebase. Baselines should be easily forkable without relying on other baselines and generic modules.
3. Prescribe best practices for training and evaluating uncertainty models.
4. Provide checkpoints for pre-trained models.

## Motivation

There are many uncertainty implementations across GitHub. However, they are typically one-off experiments for a specific project (many projects don't even have code). This raises three problems. First, there are no clear examples that uncertainty researchers can build on to quickly prototype their work. Everyone must implement their own baseline. Second, even on standard tasks such as CIFAR-10, projects differ slightly in their experiment setup, whether it be architectures, hyperparameters, or data preprocessing. This makes it difficult to compare properly across methods. Third, there is no clear guidance on which ideas and tricks necessarily contribute to getting best performance and/or are generally robust to hyperparameters.

Non-goals:

* Provide a new benchmark for uncertainty methods. Uncertainty Baselines implements many methods on already-used tasks. It does not propose new tasks. See [OpenAI Baselines](https://github.com/openai/baselines) for a work in similar spirit for RL. For new benchmarks, see [Riquelme et al. (2018)](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits), [Hendrycks and Dietterich (2019)](https://arxiv.org/abs/1903.12261), [Ovadia et al. (2019)](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019), [`OATML/bdl-benchmarks`](https://github.com/OATML/bdl-benchmarks).
