from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "breast_cancer.csv"
 
def load_dataset():
    # Load the educational dataset bundled with scikit-learn.
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    # Keep feature columns and target in one table.
    frame.to_csv(CSV_PATH, index=False)
    return frame, list(dataset.feature_names)
 
def split_dataset(test_size=0.2, random_state=42):
    # A fixed random_state makes the split repeatable.
    frame, feature_names = load_dataset()
    X = frame[feature_names]
    y = frame["target"]
    return train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
