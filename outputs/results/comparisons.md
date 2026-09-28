## Trained models vs zero-shot ASR

| Candidate | Baseline | Better on | Worse on | Tied | Mean gain (pts) | 95% CI (speakers) |
|---|---|---|---|---|---|---|
| bcresnet-1 | whisper-large-v3-lenient | 6 of 8 | 1 | 1 | +17.2 | [+6.6, +28.8] |
| bcresnet-1 | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 2 | 0 | +12.9 | [+3.7, +23.7] |
| bcresnet-2 | whisper-large-v3-lenient | 7 of 8 | 0 | 1 | +21.3 | [+10.8, +31.8] |
| bcresnet-2 | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 1 | 1 | +17.0 | [+7.6, +27.1] |
| bcresnet-3 | whisper-large-v3-lenient | 7 of 8 | 0 | 1 | +20.9 | [+9.9, +32.7] |
| bcresnet-3 | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 1 | 1 | +16.5 | [+6.7, +28.1] |
| bcresnet-8 | whisper-large-v3-lenient | 8 of 8 | 0 | 0 | +20.3 | [+11.3, +29.6] |
| bcresnet-8 | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 0 | 2 | +15.9 | [+8.6, +22.8] |
| hubert-large | whisper-large-v3-lenient | 7 of 8 | 1 | 0 | +23.0 | [+12.5, +32.6] |
| hubert-large | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 2 | 0 | +18.6 | [+8.5, +28.0] |
| hubert-base | whisper-large-v3-lenient | 5 of 8 | 3 | 0 | +14.8 | [+3.4, +27.2] |
| hubert-base | parakeet-tdt-0.6b-v3-lenient | 4 of 8 | 3 | 1 | +10.5 | [-0.0, +22.7] |
| distilhubert | whisper-large-v3-lenient | 6 of 8 | 2 | 0 | +13.6 | [+2.6, +25.3] |
| distilhubert | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 2 | 0 | +9.2 | [-0.6, +20.6] |
| hubert-large-controls | whisper-large-v3-lenient | 6 of 8 | 1 | 1 | +19.5 | [+10.2, +28.5] |
| hubert-large-controls | parakeet-tdt-0.6b-v3-lenient | 6 of 8 | 2 | 0 | +15.2 | [+6.0, +24.4] |
| hubert-base-controls | whisper-large-v3-lenient | 5 of 8 | 3 | 0 | +7.8 | [-0.6, +16.6] |
| hubert-base-controls | parakeet-tdt-0.6b-v3-lenient | 4 of 8 | 4 | 0 | +3.4 | [-4.9, +13.0] |
| distilhubert-controls | whisper-large-v3-lenient | 5 of 8 | 3 | 0 | +0.1 | [-5.7, +5.7] |
| distilhubert-controls | parakeet-tdt-0.6b-v3-lenient | 3 of 8 | 5 | 0 | -4.2 | [-9.9, +1.8] |

## Effect of dysarthric fine-tuning

| Candidate | Baseline | Better on | Worse on | Tied | Mean gain (pts) | 95% CI (speakers) |
|---|---|---|---|---|---|---|
| hubert-large | hubert-large-controls | 6 of 8 | 0 | 2 | +3.4 | [+1.6, +5.4] |
| hubert-base | hubert-base-controls | 6 of 8 | 0 | 2 | +7.1 | [+3.3, +10.9] |
| distilhubert | distilhubert-controls | 6 of 8 | 0 | 2 | +13.4 | [+6.4, +21.0] |

## Small models vs the large pretrained reference

| Candidate | Baseline | Better on | Worse on | Tied | Mean gain (pts) | 95% CI (speakers) |
|---|---|---|---|---|---|---|
| bcresnet-1 | hubert-large | 2 of 8 | 6 | 0 | -5.7 | [-10.9, -0.7] |
| bcresnet-2 | hubert-large | 4 of 8 | 4 | 0 | -1.6 | [-5.5, +2.0] |
| bcresnet-3 | hubert-large | 3 of 8 | 4 | 1 | -2.1 | [-7.2, +3.4] |
| bcresnet-8 | hubert-large | 4 of 8 | 4 | 0 | -2.7 | [-7.3, +1.6] |
