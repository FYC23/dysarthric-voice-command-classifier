| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-lenient | 1 | 66.3 | — | [53.6, 80.2] | 54.1 | 44.4 | 89.7 | 95.7 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-lenient | 1 | 70.6 | — | [59.0, 83.8] | 56.2 | 66.7 | 91.2 | 92.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| bcresnet-1 | 1 | 84.2 | — | [76.9, 91.8] | 80.3 | 77.8 | 91.7 | — | 9.5k | 4.9M | 2 | log-Mel front end not counted |
| bcresnet-2 | 1 | 87.1 | — | [82.3, 92.2] | 82.2 | 88.9 | 92.9 | — | 27.8k | 14.6M | 2 | log-Mel front end not counted |
| bcresnet-3 | 1 | 87.7 | — | [81.2, 94.2] | 84.3 | 88.9 | 91.9 | — | 54.9k | 28.9M | 2 | log-Mel front end not counted |
| bcresnet-8 | 1 | 88.4 | — | [81.9, 95.1] | 81.2 | 100.0 | 94.1 | — | 323k | 171M | 2 | log-Mel front end not counted |
