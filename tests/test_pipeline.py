import numpy as np
import pytest

from asd.data import ITEM_COLS, TARGET_COL, clean, make_synthetic, prepare
from asd.evaluate import metrics
from asd.models import build_cnn, cnn_predict_proba, ml_models, train_cnn


@pytest.fixture(scope="module")
def ds():
    return prepare(make_synthetic(n=300, seed=0), seed=0)


def test_synthetic_follows_labelling_rule():
    df = make_synthetic(n=500, seed=1)
    assert (df[ITEM_COLS].sum(axis=1) == df["Qchat-10-Score"]).all()
    assert ((df["Qchat-10-Score"] > 3) == (df[TARGET_COL] == "Yes")).all()


def test_clean_drops_leaky_columns():
    X, y = clean(make_synthetic(n=50))
    assert "Qchat-10-Score" not in X.columns and "Case_No" not in X.columns
    assert set(np.unique(y)) <= {0, 1}


def test_prepare_shapes_and_stratification(ds):
    assert ds.X_train.shape[1] == ds.X_test.shape[1] == len(ds.feature_names)
    assert ds.X_train.dtype == np.float32
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 0.05


def test_ml_models_beat_chance(ds):
    for name, model in ml_models().items():
        model.fit(ds.X_train, ds.y_train)
        acc = metrics(ds.y_test, model.predict_proba(ds.X_test)[:, 1])["accuracy"]
        assert acc > 0.8, name


def test_cnn_builds_and_trains(ds):
    model = build_cnn(ds.X_train.shape[1])
    assert model.output_shape == (None, 1)
    model, history = train_cnn(ds.X_train, ds.y_train, epochs=60)
    proba = cnn_predict_proba(model, ds.X_test)
    assert proba.shape == (len(ds.y_test),)
    assert metrics(ds.y_test, proba)["accuracy"] > 0.8
