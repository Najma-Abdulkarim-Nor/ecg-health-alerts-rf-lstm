# Data

This project uses the **ECG Heartbeat Categorization Dataset** (Fazeli, 2018). It is a preprocessed version of the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) from PhysioNet.

The CSV files are not stored in this repository. Download them from Kaggle and put them in this folder:

```
data/
├── mitbih_train.csv   # 87,554 heartbeats
└── mitbih_test.csv    # 21,892 heartbeats
```

**Option 1: browser.** Download the archive from <https://www.kaggle.com/datasets/shayanfazeli/heartbeat>, unzip it, and copy the two `mitbih_*.csv` files here.

**Option 2: Kaggle CLI** (needs a Kaggle API token):

```bash
kaggle datasets download -d shayanfazeli/heartbeat -p data --unzip
```

## Format

Each row is one heartbeat. The first 187 columns are amplitude values (normalised to the 0–1 range and zero-padded). The last column is the class label:

| Label | Class | Train samples |
|---|---|---|
| 0 | Normal (N) | 72,471 |
| 1 | Supraventricular ectopic (S) | 2,223 |
| 2 | Ventricular ectopic (V) | 5,788 |
| 3 | Fusion (F) | 641 |
| 4 | Unknown / paced (Q) | 6,431 |

In this project, labels 1–4 are merged into a single **Abnormal** class.
