"""Fetch the public source CSV locally; this project does not redistribute it."""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/kushal99-AE/UAE-Used-Car-Price-Prediction/main/dubizzle_cars_dataset.csv"
DEST = Path(__file__).parent / "data/raw/dubizzle_cars_dataset.csv"

if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, DEST)
    print(f"Downloaded {DEST}. Review data/README.md for provenance and usage limits.")
