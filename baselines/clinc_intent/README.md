# Clinc Intent Detection

## TextCNN

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.0145 / 0.2626 | 99.8% / 94.4% | 0.0093 / 0.0027 | 2.0 (8 TPUv2 cores) | 2.45M |


## BERT

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic (2 layer, 128 unit) | 0.0001 / 0.4517 | 99.9% / 94.5% | 0.0001 / 0.0381 | 1.0 (8 TPUv3 cores) | 4M |
| Deterministic (4 layer, 256 unit) | 0.0001 / 0.2796 | 99.9% / 96.8% | 0.0001 / 0.0286 | 1.0 (8 TPUv3 cores) | 11M |
| Deterministic (8 layer, 512 unit) | 0.0002 / 0.3617 | 99.9% / 97.6% | 0.0001 / 0.0256 | 1.0 (8 TPUv3 cores) | 41M |
| Deterministic (12 layer, 768 unit) | 0.0002 / 0.1854 | 99.9% / 97.7% | 0.0001 / 0.0187 | 1.0 (8 TPUv3 cores) | 110M |
| Deterministic (24 layer, 1024 unit) | 0.0001 / 0.1402 | 99.9% / 98.1% | 0.0001 / 0.0236 | 1.0 (8 TPUv3 cores) | 340M |
