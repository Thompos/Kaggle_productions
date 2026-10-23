from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def save_processed_data(train, test, clean_all):
    train.to_pickle(PROCESSED_DIR / "train_clean.pkl")
    test.to_pickle(PROCESSED_DIR / "test_clean.pkl")
    clean_all.to_pickle(PROCESSED_DIR / "clean_ALL.pkl")


def load_processed_data():
    train = pd.read_pickle(PROCESSED_DIR / "train_clean.pkl")
    test = pd.read_pickle(PROCESSED_DIR / "test_clean.pkl")
    clean_all = pd.read_pickle(PROCESSED_DIR / "clean_ALL.pkl")
    return train, test, clean_all
