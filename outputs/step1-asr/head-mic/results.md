| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-strict | 1 | 65.1 | — | [47.9, 80.6] | 47.2 | 71.0 | 87.0 | 94.7 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| whisper-large-v3-lenient | 1 | 75.4 | — | [64.4, 86.2] | 60.9 | 83.9 | 91.9 | 97.9 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-strict | 1 | 39.4 | — | [23.2, 57.8] | 23.6 | 12.9 | 69.2 | 80.5 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| parakeet-tdt-0.6b-v3-lenient | 1 | 53.3 | — | [37.2, 70.0] | 43.3 | 16.1 | 79.1 | 90.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
