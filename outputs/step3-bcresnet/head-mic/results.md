| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-lenient | 1 | 75.4 | — | [64.4, 86.2] | 60.9 | 83.9 | 91.9 | 97.9 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-lenient | 1 | 53.3 | — | [37.2, 70.0] | 43.3 | 16.1 | 79.1 | 90.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| bcresnet-1 | 1 | 86.3 | — | [80.9, 92.5] | 81.9 | 80.6 | 94.1 | — | 9.5k | 4.9M | 2 | log-Mel front end not counted |
| bcresnet-2 | 1 | 87.5 | — | [81.1, 93.3] | 81.2 | 93.5 | 93.9 | — | 27.8k | 14.6M | 2 | log-Mel front end not counted |
| bcresnet-3 | 1 | 90.3 | — | [85.3, 95.2] | 84.4 | 96.8 | 96.1 | — | 54.9k | 28.9M | 2 | log-Mel front end not counted |
| bcresnet-8 | 1 | 92.4 | — | [86.2, 97.7] | 87.9 | 93.5 | 98.0 | — | 323k | 171M | 2 | log-Mel front end not counted |
