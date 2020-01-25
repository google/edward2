# ResNet-50 on ImageNet

| Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Top-5 Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| BatchEnsemble<sup>1</sup> | 0.847 / 0.951 | 79.2% / 76.5% | - | 17.5 (32 TPUv2 cores) | 25.8M |
| Deterministic | 0.907 / 0.935 | 77.9% / 76.1% | - | 5 (32 TPUv3 cores) | 25.6M |

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Top-5 Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`facebookarchive/fb.resnet.torch `](https://github.com/facebookarchive/fb.resnet.torch ) | Deterministic | - | - / 75.99% | - / 92.98% | - | 25.6M |
| [`KaimingHe/deep-residual-networks`](https://github.com/KaimingHe/deep-residual-networks) | Deterministic | - | - / 75.3% | - | - | 25.6M |
| [`keras-team/keras`](https://keras.io/applications/#resnet) | Deterministic | - | - / 74.9% | - / 92.1% | - | 25.6M |
| | Deterministic (ResNet-152v2) | - | - / 78.0% | - / 94.2% | - | 60.3M |
| [`tensorflow/tpu`](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)<sup>2</sup> | Deterministic | - | - / 76% | - | 17 (8 TPUv2) | 25.6M |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>3</sup> | Adaptive Thermostat Monte Carlo (single sample) | - / 1.08 | - / 74.2% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.883 | - / 77.5% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 1.15 | - / 73.1% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.941 | - / 76.4% | - | 1000 epochs (8 TPUv3 cores) | - |
| [Maddox et al. (2019)](https://arxiv.org/abs/1902.02476)<sup>4</sup> | Deterministic (ResNet-152) | - / 0.8716 | - / 78.39% | - | pretrained+10 epochs | 60.3M |
| | SWA | - / 0.8682 | - / 78.92% | - | pretrained+10 epochs | 60.3M |
| | SWAG | - / 0.8205 | - / 79.08% | - | pretrained+10 epochs | 1.33B |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>5</sup>  | Variational Online Gauss-Newton | - / 1.37 | 73.87% / 67.38% | | 1.90 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530) | Deterministic | - | - | - | - | - |
| | Dropout | - | - | - | - | - |
| | Ensemble | - | - | - | - | - |
| [Zhang et al. (2019)](https://openreview.net/forum?id=rkeS1RVtPS)<sup>6</sup> | Deterministic (ResNet-50) | - / 0.960 | - / 76.046% | - /  92.78% | 25.6M |
| | cSGHMC | - / 0.888 | - / 77.11% | - / 93.524% | 307.2M |

1. Each ensemble member achieves roughly 75.6 test top-1 accuracy.
2. See documentation for differences from original paper, e.g., preprocessing.
3. Modifies architecture. Cyclical learning rate.
4. Uses ResNet-152. Training uses pre-trained SGD solutions. SWAG uses rank 20 which requires 20 + 2 copies of the model parameters, and 30 samples at test time.
5. Uses ResNet-18. Scales KL by an additional factor of 5.
6. cSGHMC uses a total of 9 copies of the full size of weights for prediction. The authors use a T=1/200 temperature scaling on the log-posterior (see the newly added appendix I at https://openreview.net/forum?id=rkeS1RVtPS)

TODO(trandustin): Add column for Test runtime.

TODO(trandustin): Add column for Checkpoints.
