## Trained models vs zero-shot ASR

| Candidate | Baseline | Better on | Worse on | Tied | Mean gain (pts) | 95% CI (speakers) |
|---|---|---|---|---|---|---|
| bcresnet-1 | whisper-large-v3-lenient | 6 of 8 | 2 | 0 | +12.1 | [+3.6, +21.7] |
| bcresnet-1 | parakeet-tdt-0.6b-v3-lenient | 8 of 8 | 0 | 0 | +34.2 | [+20.4, +47.9] |
| bcresnet-2 | whisper-large-v3-lenient | 7 of 8 | 1 | 0 | +15.3 | [+7.1, +24.3] |
| bcresnet-2 | parakeet-tdt-0.6b-v3-lenient | 8 of 8 | 0 | 0 | +37.4 | [+22.4, +53.1] |
| bcresnet-3 | whisper-large-v3-lenient | 7 of 8 | 0 | 1 | +15.6 | [+8.9, +22.5] |
| bcresnet-3 | parakeet-tdt-0.6b-v3-lenient | 8 of 8 | 0 | 0 | +37.6 | [+23.3, +53.2] |
| bcresnet-8 | whisper-large-v3-lenient | 7 of 8 | 0 | 1 | +14.8 | [+7.9, +22.2] |
| bcresnet-8 | parakeet-tdt-0.6b-v3-lenient | 8 of 8 | 0 | 0 | +36.9 | [+23.5, +50.9] |
| hubert-large | whisper-large-v3-lenient | 7 of 8 | 1 | 0 | +18.0 | [+8.5, +28.1] |
| hubert-large | parakeet-tdt-0.6b-v3-lenient | 7 of 8 | 0 | 1 | +40.0 | [+24.9, +54.9] |
| hubert-base | whisper-large-v3-lenient | 6 of 8 | 2 | 0 | +12.7 | [+4.7, +20.8] |
| hubert-base | parakeet-tdt-0.6b-v3-lenient | 7 of 8 | 1 | 0 | +34.8 | [+20.4, +48.9] |
| distilhubert | whisper-large-v3-lenient | 6 of 8 | 2 | 0 | +6.0 | [-1.0, +13.2] |
| distilhubert | parakeet-tdt-0.6b-v3-lenient | 7 of 8 | 1 | 0 | +28.1 | [+15.9, +40.6] |
| hubert-large-controls | whisper-large-v3-lenient | 6 of 8 | 1 | 1 | +12.8 | [+4.4, +21.5] |
| hubert-large-controls | parakeet-tdt-0.6b-v3-lenient | 7 of 8 | 1 | 0 | +34.8 | [+20.6, +48.8] |
| hubert-base-controls | whisper-large-v3-lenient | 6 of 8 | 2 | 0 | +5.4 | [-1.6, +11.6] |
| hubert-base-controls | parakeet-tdt-0.6b-v3-lenient | 8 of 8 | 0 | 0 | +27.4 | [+16.4, +38.5] |
| distilhubert-controls | whisper-large-v3-lenient | 3 of 8 | 3 | 2 | -2.7 | [-9.4, +3.4] |
| distilhubert-controls | parakeet-tdt-0.6b-v3-lenient | 7 of 8 | 1 | 0 | +19.3 | [+8.9, +29.6] |

## Effect of dysarthric fine-tuning

| Candidate | Baseline | Better on | Worse on | Tied | Mean gain (pts) | 95% CI (speakers) |
|---|---|---|---|---|---|---|
| hubert-large | hubert-large-controls | 6 of 8 | 1 | 1 | +5.2 | [+2.2, +8.4] |
| hubert-base | hubert-base-controls | 6 of 8 | 1 | 1 | +7.3 | [+3.0, +11.8] |
| distilhubert | distilhubert-controls | 7 of 8 | 0 | 1 | +8.8 | [+4.0, +13.9] |

## Small models vs the large pretrained reference

| Candidate | Baseline | Better on | Worse on | Tied | Mean gain (pts) | 95% CI (speakers) |
|---|---|---|---|---|---|---|
| bcresnet-1 | hubert-large | 1 of 8 | 6 | 1 | -5.8 | [-9.0, -2.7] |
| bcresnet-2 | hubert-large | 2 of 8 | 4 | 2 | -2.6 | [-5.9, +0.2] |
| bcresnet-3 | hubert-large | 2 of 8 | 4 | 2 | -2.4 | [-6.4, +1.3] |
| bcresnet-8 | hubert-large | 2 of 8 | 5 | 1 | -3.1 | [-6.7, +0.1] |
