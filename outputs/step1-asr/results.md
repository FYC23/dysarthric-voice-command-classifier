| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-strict | 1 | 50.3 | — | [33.4, 69.2] | 34.2 | 22.2 | 81.3 | 92.1 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| whisper-large-v3-lenient | 1 | 66.3 | — | [53.6, 80.2] | 54.1 | 44.4 | 89.7 | 95.7 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-strict | 1 | 53.9 | — | [38.8, 70.6] | 36.4 | 44.4 | 80.3 | 82.3 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| parakeet-tdt-0.6b-v3-lenient | 1 | 70.6 | — | [59.0, 83.8] | 56.2 | 66.7 | 91.2 | 92.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
