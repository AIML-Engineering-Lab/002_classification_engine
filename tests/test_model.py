"""Tests for Classification Engine model."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_model_files_exist():
    assert (ROOT / "models" / "logreg_manuscript.pkl").exists(), "Manuscript model not found."
    assert (ROOT / "models" / "logreg_silicon.pkl").exists(), "Silicon model not found."


def test_prediction_output():
    from predict import predict
    # Manuscript
    df_m = pd.read_csv(ROOT / "data" / "manuscript_authenticity_data.csv")
    feats_m = df_m.drop(columns=["is_authentic"]).head(3)
    preds_m = predict(feats_m, str(ROOT / "models" / "logreg_manuscript.pkl"))
    assert len(preds_m) == 3

    # Silicon
    df_s = pd.read_csv(ROOT / "data" / "silicon_timing_test_data.csv")
    feats_s = df_s.drop(columns=["timing_pass"]).head(3)
    preds_s = predict(feats_s, str(ROOT / "models" / "logreg_silicon.pkl"))
    assert len(preds_s) == 3


if __name__ == "__main__":
    test_model_files_exist()
    test_prediction_output()
    print("All tests passed.")
