| Model | Seeds | Acc. % | ± seed std | 95% CI (speakers) | Severe | Mod.-severe | Mild | Control | #Params | #MACs | Input s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-lenient | 1 | 75.4 | — | [64.4, 86.2] | 60.9 | 83.9 | 91.9 | 97.9 | 1.5G | 1.3T | 2 | encoder padded to 30 s; greedy decode of one clip |
| parakeet-tdt-0.6b-v3-lenient | 1 | 53.3 | — | [37.2, 70.0] | 43.3 | 16.1 | 79.1 | 90.0 | 627M | 16.7G | 2 | greedy TDT decode of one clip; multilingual, language not forced |
| bcresnet-1 | 3 | 87.5 | 1.0 | [82.0, 93.0] | 83.9 | 82.8 | 93.9 | — | 9.5k | 4.9M | 2 | log-Mel front end not counted |
| bcresnet-2 | 3 | 90.7 | 2.9 | [85.9, 95.4] | 87.1 | 93.5 | 94.7 | — | 27.8k | 14.6M | 2 | log-Mel front end not counted |
| bcresnet-3 | 3 | 91.0 | 0.6 | [85.5, 95.9] | 85.2 | 96.8 | 96.7 | — | 54.9k | 28.9M | 2 | log-Mel front end not counted |
| bcresnet-8 | 3 | 90.2 | 2.7 | [84.8, 95.4] | 85.3 | 90.3 | 96.7 | — | 323k | 171M | 2 | log-Mel front end not counted |
| hubert-large | 3 | 93.4 | 0.5 | [88.5, 97.2] | 90.5 | 91.4 | 97.8 | — | 316M | 36.2G | 2 | MACs include the CNN front end, every transformer layer and the head |
| hubert-base | 3 | 88.1 | 2.0 | [82.3, 93.2] | 84.4 | 82.8 | 94.8 | — | 94.8M | 14.0G | 2 | MACs include the CNN front end, every transformer layer and the head |
| distilhubert | 3 | 81.4 | 3.3 | [74.3, 88.0] | 75.2 | 76.3 | 91.5 | — | 23.9M | 6.9G | 2 | MACs include the CNN front end, every transformer layer and the head |
| hubert-large-controls | 3 | 88.2 | 2.5 | [81.6, 94.4] | 82.4 | 83.9 | 97.3 | — | 316M | 36.2G | 2 | trained on control speakers only; MACs include the CNN front end, every transformer layer and the head |
| hubert-base-controls | 3 | 80.8 | 0.6 | [71.6, 89.7] | 72.8 | 69.9 | 94.9 | — | 94.8M | 14.0G | 2 | trained on control speakers only; MACs include the CNN front end, every transformer layer and the head |
| distilhubert-controls | 3 | 72.6 | 5.5 | [61.9, 83.2] | 62.3 | 63.4 | 89.5 | — | 23.9M | 6.9G | 2 | trained on control speakers only; MACs include the CNN front end, every transformer layer and the head |
