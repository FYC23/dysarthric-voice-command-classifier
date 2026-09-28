| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-lenient | 1 | 66.3 | — | [53.6, 80.2] | 54.1 | 44.4 | 89.7 | 95.7 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-lenient | 1 | 70.6 | — | [59.0, 83.8] | 56.2 | 66.7 | 91.2 | 92.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| bcresnet-1 | 2 | 83.5 | 1.0 | [76.6, 90.0] | 78.2 | 83.3 | 90.6 | — | 9.5k | 4.9M | 2 | log-Mel front end not counted |
| bcresnet-2 | 2 | 87.6 | 0.8 | [82.9, 92.7] | 84.4 | 83.3 | 93.4 | — | 27.8k | 14.6M | 2 | log-Mel front end not counted |
| bcresnet-3 | 2 | 87.2 | 0.8 | [81.2, 93.7] | 83.9 | 83.3 | 92.9 | — | 54.9k | 28.9M | 2 | log-Mel front end not counted |
| bcresnet-8 | 2 | 86.5 | 2.6 | [80.8, 92.6] | 79.1 | 88.9 | 95.6 | — | 323k | 171M | 2 | log-Mel front end not counted |
