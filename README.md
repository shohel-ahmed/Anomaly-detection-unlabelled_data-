# Data Validation  
## Anomaly Detection from Unlabelled Data

A lightweight Python module for **unsupervised anomaly detection** on tabular data using **Isolation Forest**.  
It’s designed to work efficiently with large CSV/Parquet files via **DuckDB sampling**.  
It also includes a robust preprocessing pipeline (datetime handling, data imputation, categorical encoding, scaling) and generates explainability outputs (SHAP, PCA plots, tables).

---

## Requirements

### Python
- **Version:** 3.9–3.11 (recommended)

### Dependencies
(as imported in the module)
- `duckdb`
- `numpy`
- `pandas`
- `scikit-learn`
- `shap`
- `matplotlib`
- `colorama`

---

## 1) Prepare Data

- Place either `your_table.csv` or `your_table.parquet` inside a folder (e.g., `./data`).
- The module automatically detects the file type and loads it using DuckDB.

### Notes
- Samples **up to 500,000 rows** (or fewer if the dataset is smaller).
- Supported types:
  - numeric  
  - categorical (`object` / `category`)  
  - datetime-like columns (converted to **epoch seconds**)
- Extremely high-cardinality string ID columns are automatically **dropped**.

---

## 2) Run from a Small Script

```python
from data_validation_ISO import MLAnomalyDetection

if __name__ == "__main__":
    # Your data location path
    target_path = "./data"

    # File base name (without extension): expects `your_table.csv` or `your_table.parquet`
    file_name = "your_table"

    # Optional: sample size for Isolation Forest (stability on very large data)
    iforest_max_samples = 100000

    anomaly_obj = MLAnomalyDetection(
        target_path=target_path,
        file_name=file_name,
        iforest_max_samples=iforest_max_samples
    )
    anomaly_obj.execute()

