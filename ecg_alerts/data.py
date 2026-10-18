"""Loading, binarising and rebalancing the MIT-BIH heartbeat dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

SEQ_LEN = 187

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)",
}


def load_mitbih(data_dir="data"):
    """Load the Kaggle MIT-BIH train/test CSVs.

    Each row holds 187 amplitude values followed by the class label (0-4).
    Returns X_train, y_train_raw, X_test, y_test_raw.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "mitbih_train.csv"
    test_path = data_dir / "mitbih_test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the dataset as described in data/README.md."
            )

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    X_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train_raw = train_df.iloc[:, -1].to_numpy().astype(int)
    y_test_raw = test_df.iloc[:, -1].to_numpy().astype(int)
    return X_train, y_train_raw, X_test, y_test_raw


def to_binary(y_raw):
    """Map class 0 to Normal (0) and classes 1-4 to Abnormal (1)."""
    return (np.asarray(y_raw) > 0).astype(int)


def balance_with_smote(X, y, k_neighbors=5, random_state=42):
    """Oversample the minority class with SMOTE. Apply to training data only."""
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    return smote.fit_resample(X, y)
