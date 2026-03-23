import pandas as pd
from pathlib import Path

DATA = Path("data/raw")

for f in DATA.glob("*.csv"):
    df = pd.read_csv(f, low_memory=False)
    print(f.name)
    print("shape:", df.shape)
    print("first columns:", list(df.columns[:10]))
    print()

