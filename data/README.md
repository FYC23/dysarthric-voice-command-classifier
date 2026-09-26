# Data

Everything in this folder except this README and `labels/` is gitignored. Code finds it
through the paths in `src/config.py`, all relative to the repo root.

```
data/
├── raw/                      # Datasets exactly as downloaded. Never modified.
│   ├── TORGO/
│   └── speech_commands_v2/
├── cache/                    # Derived files. Safe to delete; scripts rebuild them.
│   └── pretrained/           # Downloaded pretrained weights (HuBERT)
└── labels/                   # Hand labels. Committed: they can't be regenerated.
    └── torgo_manual_segment_labels.json  # Hand-labelled word / partial / non-speech
                                          # segments in the 36 longest TORGO clips

runs/                         # (repo root) training checkpoints, also gitignored
```

## Setup

`data/raw/` and `data/cache/` are plain folders inside the repo. Each dataset
has a download script that fills its folder:

```bash
bash scripts/download_torgo.sh
bash scripts/download_speech_commands.sh
```

`data/` itself must stay a real folder, not a symlink, or git cannot track this README.

## TORGO

From the [TORGO database page](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
(University of Toronto). The script fetches the four archives (`F`, `FC`, `M`,
`MC`, about 9.6 GB compressed) from that server and extracts each into its own
group folder, since the archives contain only the speaker folders. It skips
groups that are already extracted.

If `<group>.tar.bz2` is already in `data/raw/TORGO/` (e.g. downloaded in a
browser), the script extracts it from disk and leaves the archive in place.
Otherwise it streams the download straight into `tar`. Set `TORGO_BASE_URL` to
use a mirror.

```
data/raw/TORGO/
├── F/  FC/  M/  MC/              # female, female control, male, male control
│   └── F01/
│       └── Session1/
│           ├── prompts/0001.txt  # the word that was spoken
│           └── wav_arrayMic/0001.wav
```

`src/data/preprocessing.py` scans this layout and reads `wav_arrayMic` by
default (`Config.MIC_TYPE`).

## Speech Commands v0.02

Google Speech Commands v0.02 ([Warden, 2018](https://arxiv.org/abs/1804.03209)),
with the same train/validation/test split as
[`google/speech_commands`](https://huggingface.co/datasets/google/speech_commands)
on Hugging Face. That repo contains only a loading script, so the download
script fetches the S3 archives it points to and extracts them directly:

```bash
bash scripts/download_speech_commands.sh
```

About 2.2 GB to download and 3.0 GB extracted. The script streams each archive
straight into `tar`, so no `.tar.gz` stays on disk. It skips any split that is
already present.

```
data/raw/speech_commands_v2/
├── train/        # 84,848 clips
├── validation/   #  9,982 clips
└── test/         #  4,890 clips
    ├── yes/0a7c2a8d_nohash_0.wav   # <word>/<speaker>_nohash_<n>.wav
    ├── ...                         # 35 words
    └── _silence_/                  # background-noise recordings, not 1 s clips
```

Clips are 1 s, 16 kHz, mono int16, matching `Config.SAMPLE_RATE`.

It covers 17 of the 20 target commands: all ten digits plus `yes`, `no`, `up`,
`down`, `left`, `right` and `forward`. `back`, `select` and `menu` are missing
(the nearest word is `backward`).
