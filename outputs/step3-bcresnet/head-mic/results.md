| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-lenient | 1 | 75.4 | — | [64.4, 86.2] | 60.9 | 83.9 | 91.9 | 97.9 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-lenient | 1 | 53.3 | — | [37.2, 70.0] | 43.3 | 16.1 | 79.1 | 90.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| bcresnet-1 | 2 | 87.2 | 1.2 | [82.3, 92.5] | 82.9 | 80.6 | 95.0 | — | 9.5k | 4.9M | 2 | log-Mel front end not counted |
| bcresnet-2 | 2 | 89.6 | 2.9 | [84.0, 94.7] | 84.9 | 93.5 | 94.5 | — | 27.8k | 14.6M | 2 | log-Mel front end not counted |
| bcresnet-3 | 2 | 91.0 | 0.9 | [86.1, 95.6] | 84.9 | 96.8 | 97.1 | — | 54.9k | 28.9M | 2 | log-Mel front end not counted |
| bcresnet-8 | 2 | 89.8 | 3.7 | [84.0, 95.1] | 84.3 | 90.3 | 97.1 | — | 323k | 171M | 2 | log-Mel front end not counted |
