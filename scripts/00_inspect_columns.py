import pandas as pd
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("data/raw") 

# The files we need to check
files = [
    "BSHRI_PLA_CSF_NULISA_CNS_20Jun2026.csv",
    "DXSUM_20Jun2026.csv",
    "PTDEMOG_20Jun2026.csv",
    "APOERES_20Jun2026.csv",
    "MMSE_20Jun2026.csv",
]

print("--- SAFE COLUMN INSPECTION ---")
for f in files:
    try:
        # Read only the header (0 rows)
        df = pd.read_csv(DATA_DIR / f, nrows=0)
        print(f"\nFILE: {f}")
        print(f"COLUMNS: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading {f}: {e}")