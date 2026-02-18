import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    # FIX WILAYAH COLUMN
    if "wilayah" not in df.columns:
        if "name" in df.columns:
            df["wilayah"] = df["name"]
        else:
            df["wilayah"] = df["provinsi"]

    df = df[df["tanggal"].notna()]
    df = df[df["harga"].notna()]
    df = df[df["wilayah"].notna()]

    return df
