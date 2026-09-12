"""Data loading and subsampling utilities.

Handles downloading the Kaggle dataset and loading it into pandas DataFrames.
"""

import os
import hashlib
import pandas as pd
import kagglehub
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def download_dataset(force: bool = False) -> Path:
    """Download the Kaggle customer-support-on-twitter dataset.

    Args:
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded CSV file.
    """
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "twcs.csv"

    if csv_path.exists() and not force:
        print(f"[DATA] Dataset already exists at {csv_path}")
        return csv_path

    print("[DATA] Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    # kagglehub downloads to a cache dir; copy to our data/ dir
    import shutil
    src = Path(path) / "twcs.csv"
    if src.exists():
        shutil.copy2(src, csv_path)
        print(f"[DATA] Copied to {csv_path}")
    else:
        # Find any CSV in the downloaded dir
        csvs = list(Path(path).glob("*.csv"))
        if csvs:
            shutil.copy2(csvs[0], csv_path)
            print(f"[DATA] Copied {csvs[0].name} to {csv_path}")
        else:
            raise FileNotFoundError(f"No CSV found in downloaded dataset at {path}")

    return csv_path


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load the dataset from CSV.

    Args:
        csv_path: Path to the CSV file. Defaults to data/twcs.csv.

    Returns:
        DataFrame with the full dataset.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "twcs.csv"
    print(f"[DATA] Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[DATA] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def subsample(
    df: pd.DataFrame,
    n_samples: int = 30000,
    seed: int = 42,
    method: str = "stratified",
) -> pd.DataFrame:
    """Subsample the dataset for faster exploration.

    Args:
        df: Full DataFrame.
        n_samples: Number of rows to sample.
        seed: Random seed for reproducibility.
        method: Sampling method - "stratified" (by author_id) or "random".

    Returns:
        Subsampled DataFrame.
    """
    if len(df) <= n_samples:
        print(f"[DATA] Dataset already smaller than n_samples={n_samples}, returning full data")
        return df

    if method == "stratified":
        # Stratify by author_id to preserve brand distribution
        sampled = df.groupby("author_id", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(1, int(n_samples * len(x) / len(df)))),
                random_state=seed,
            )
        )
        # If over-sampled due to groupby, trim to exact n_samples
        if len(sampled) > n_samples:
            sampled = sampled.sample(n=n_samples, random_state=seed)
    else:
        sampled = df.sample(n=n_samples, random_state=seed)

    print(f"[DATA] Subsampled to {len(sampled):,} rows (method={method}, seed={seed})")
    return sampled.reset_index(drop=True)


def get_sample_hash(df: pd.DataFrame) -> str:
    """Get a short hash of the sample for reproducibility tracking."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:8]
