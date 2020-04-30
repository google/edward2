# Clinc Intent Detection

## TextCNN

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.0145 / 0.2626 | 99.8% / 94.4% | 0.0093 / 0.0027 | 2.0 (8 TPUv2 cores) | 2.45M |


## BERT

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic (2 layer, 128 unit) | 0.0001 / 0.4927 | 99.9% / 92.9% | 0.0001 / 0.0394 | 1.0 (8 TPUv3 cores) | 4.40M |
| Deterministic (4 layer, 256 unit) | 0.0001 / 0.4296 | 99.9% / 95.3% | 0.0001 / 0.0398 | 1.0 (8 TPUv3 cores) | 11.21M |
| Deterministic (8 layer, 512 unit) | 0.0002 / 0.3617 | 99.9% / 95.7% | 0.0001 / 0.0407 | 1.0 (8 TPUv3 cores) | 41.45M |
| Deterministic (12 layer, 768 unit) | 0.0002 / 0.6926 | 99.9% / 94.9% | 0.0001 / 0.0495 | 1.0 (8 TPUv3 cores) | 109.60M |
