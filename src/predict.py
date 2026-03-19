"""
Inference for Classification Engine.
Load trained model and run predictions on new data.
"""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    """Load model and predict on input DataFrame."""
    if model_path is None:
        model_path = str(MODEL_DIR / "logreg_silicon.pkl")

    pipe = joblib.load(model_path)
    X = data.select_dtypes(include="number")
    preds = pipe.predict(X)
    return preds.tolist()


if __name__ == "__main__":
    # Demo: Manuscript predictions
    df_m = pd.read_csv(ROOT / "data" / "manuscript_authenticity_data.csv")
    feats_m = df_m.drop(columns=["is_authentic"]).head(5)
    preds_m = predict(feats_m, str(MODEL_DIR / "logreg_manuscript.pkl"))
    print(f"Manuscript predictions: {preds_m}")

    # Demo: Silicon predictions
    df_s = pd.read_csv(ROOT / "data" / "silicon_timing_test_data.csv")
    feats_s = df_s.drop(columns=["timing_pass"]).head(5)
    preds_s = predict(feats_s, str(MODEL_DIR / "logreg_silicon.pkl"))
    print(f"Silicon predictions:    {preds_s}")
