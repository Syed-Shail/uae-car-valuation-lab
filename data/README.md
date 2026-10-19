# Data access and provenance

The project uses a public UAE/Dubizzle used-car listing snapshot distributed in the GitHub repository below. The upstream README describes vehicle listing attributes but does not state a dataset license. For that reason, this repository does **not** redistribute the CSV; fetch it yourself and review the upstream terms before reuse or publication.

- Upstream project: <https://github.com/kushal99-AE/UAE-Used-Car-Price-Prediction>
- Source file: `dubizzle_cars_dataset.csv`
- Direct download: <https://raw.githubusercontent.com/kushal99-AE/UAE-Used-Car-Price-Prediction/main/dubizzle_cars_dataset.csv>
- Local expected path: `data/raw/dubizzle_cars_dataset.csv`
- Snapshot downloaded for this analysis: 2026-10-19
- SHA-256 of downloaded CSV: `ebeacfdbcf20c410cf94317f24ac0452f7ce7d86127d67f5cc52d512f2188980`
- Rows and columns in that snapshot: 9,953 × 27

The source describes the data as used-car listings in the UAE. It does not provide listing timestamps, sale outcomes, or a collection methodology. The model therefore estimates **asking prices represented in this historical snapshot**, not realized transaction prices or today's market value. Its test score measures prediction on a held-out portion of the same snapshot, not future-market performance.

The upstream notebook's price-cleaning section sets `price_threshold = 5000` and says entries below that level are wrong or fake. This project follows that source-documented screen before splitting the data; the threshold and excluded row count are included in `artifacts/results.json`. See the [upstream analysis notebook](https://github.com/kushal99-AE/UAE-Used-Car-Price-Prediction/blob/main/Car_Price_Prediction.ipynb).

The downloadable file URL points to a mutable `main` branch. After fetching, record the SHA-256 digest in your experiment notes so the exact input can be identified. Do not treat the data as current market data.
