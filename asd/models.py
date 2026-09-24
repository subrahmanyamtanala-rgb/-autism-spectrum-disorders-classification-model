"""Classical machine-learning baselines and a 1D CNN for tabular screening data."""

from __future__ import annotations

import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def ml_models(seed: int = 42) -> dict:
    """The classical classifiers compared against the CNN."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        # Platt-scaled SVM so it exposes predict_proba like the other models.
        "SVM (RBF)": CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=seed), ensemble=False),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Naive Bayes": GaussianNB(),
    }


def build_cnn(n_features: int, learning_rate: float = 1e-3):
    """1D CNN that treats the encoded feature vector as a length-n sequence.

    Two Conv1D blocks learn local interactions between neighbouring
    questionnaire items (A1..A10 are adjacent in the vector). The feature maps
    are flattened rather than globally pooled so the position of each item is
    kept, followed by a small dense head with dropout for regularisation.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(n_features, 1)),
            layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="asd_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_cnn(X_train, y_train, epochs: int = 150, batch_size: int = 32, seed: int = 42, verbose: int = 0):
    """Train the CNN with a stratified 15% validation split, early stopping and deterministic ops."""
    import tensorflow as tf

    from sklearn.model_selection import train_test_split

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    # Stratified hold-out so the validation set keeps the class ratio.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed)
    model = build_cnn(X_train.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    history = model.fit(
        X_fit[..., np.newaxis], y_fit,
        validation_data=(X_val[..., np.newaxis], y_val), epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=verbose,
    )
    return model, history.history


def cnn_predict_proba(model, X) -> np.ndarray:
    return model.predict(np.asarray(X, dtype="float32")[..., np.newaxis], verbose=0).ravel()
