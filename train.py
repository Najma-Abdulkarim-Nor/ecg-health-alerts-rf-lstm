"""Train and evaluate Random Forest and LSTM ECG classifiers.

Examples:
    python train.py                   # both models
    python train.py --skip-lstm       # Random Forest only (fast, no TensorFlow needed)
    python train.py --epochs 10
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from ecg_alerts.data import SEQ_LEN, balance_with_smote, load_mitbih, to_binary
from ecg_alerts.evaluate import compute_metrics, format_metrics
from ecg_alerts.features import extract_statistical_features
from ecg_alerts.models import build_lstm, build_random_forest, lstm_callbacks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder containing mitbih_train.csv and mitbih_test.csv")
    p.add_argument("--output-dir", default="results", help="where metrics.json and models/ are written")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5, help="LSTM decision threshold")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--no-smote", action="store_true", help="train on the original imbalanced data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    X_train, y_train_raw, X_test, y_test_raw = load_mitbih(args.data_dir)
    y_train, y_test = to_binary(y_train_raw), to_binary(y_test_raw)
    print(f"Train {X_train.shape}  Test {X_test.shape}  "
          f"Abnormal share (train) {y_train.mean()*100:.1f}%")

    if args.no_smote:
        X_bal, y_bal = X_train, y_train
    else:
        X_bal, y_bal = balance_with_smote(X_train, y_train, random_state=args.seed)
        print(f"After SMOTE: {len(y_bal):,} samples ({(y_bal == 0).sum():,} per class)")

    results = {}

    # Random Forest on 13 statistical features
    scaler = StandardScaler()
    Xtr_rf = scaler.fit_transform(extract_statistical_features(X_bal))
    Xte_rf = scaler.transform(extract_statistical_features(X_test))

    rf = build_random_forest(random_state=args.seed)
    t0 = time.time()
    rf.fit(Xtr_rf, y_bal)
    rf_train_time = time.time() - t0
    t0 = time.time()
    rf_pred = rf.predict(Xte_rf)
    rf_infer_time = time.time() - t0
    rf_prob = rf.predict_proba(Xte_rf)[:, 1]
    results["random_forest"] = compute_metrics(y_test, rf_pred, rf_prob)
    results["random_forest"].update(train_time_s=rf_train_time, inference_time_s=rf_infer_time)
    print(format_metrics("Random Forest", results["random_forest"]))

    joblib.dump(rf, out_dir / "models" / "random_forest.joblib")
    joblib.dump(scaler, out_dir / "models" / "feature_scaler.joblib")

    # LSTM on raw 187-point sequences
    if not args.skip_lstm:
        import tensorflow as tf

        tf.random.set_seed(args.seed)
        lstm = build_lstm()
        t0 = time.time()
        lstm.fit(
            X_bal.reshape(-1, SEQ_LEN, 1), y_bal,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.15,
            callbacks=lstm_callbacks(),
            verbose=1,
        )
        lstm_train_time = time.time() - t0
        t0 = time.time()
        lstm_prob = lstm.predict(X_test.reshape(-1, SEQ_LEN, 1), verbose=0).ravel()
        lstm_infer_time = time.time() - t0
        results["lstm"] = compute_metrics(y_test, (lstm_prob >= args.threshold).astype(int), lstm_prob)
        results["lstm"].update(train_time_s=lstm_train_time, inference_time_s=lstm_infer_time,
                               threshold=args.threshold)
        print(format_metrics("LSTM", results["lstm"]))
        lstm.save(out_dir / "models" / "lstm.keras")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
