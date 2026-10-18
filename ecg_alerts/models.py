"""Model definitions. TensorFlow is imported lazily so the RF path runs without it."""

from sklearn.ensemble import RandomForestClassifier

from .data import SEQ_LEN


def build_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_lstm(learning_rate=1e-3):
    """Two stacked LSTM layers (64 -> 32) with BatchNorm and Dropout, sigmoid output."""
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(SEQ_LEN, 1)),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def lstm_callbacks():
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
