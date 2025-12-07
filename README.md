# Fairness-Aware Income Prediction

This project investigates fairness issues in binary income prediction on the Adult (Census Income) dataset and evaluates mitigation techniques that improve group fairness without discarding predictive performance. It was developed for the *Trustworthy Machine Learning 2025 – Task 3 (Fairness-aware ML)* course project.

---

## Project Goals
- Quantify discrimination present in the raw dataset and in standard supervised models.
- Train baseline classifiers (Logistic Regression, Random Forest, XGBoost) and report both performance and fairness metrics.
- Apply post-processing Equal Opportunity adjustments to reduce disparity between privileged and unprivileged groups.
- Provide transparent, reproducible experiments for studying the trade-off between accuracy and fairness.

---

## Dataset
- **Source:** UCI Adult (Census Income).
- **Size:** ~50k individuals with demographic attributes and binary income label (`>50K` / `<=50K`).
- **Protected attributes:**
  - Primary: `sex` (Male / Female)
  - Secondary (derived): `race_binary` (White / Non-White)
- **Processed files:**
  - `data/processed/adult/adult_clean.csv` – cleaned dataset after handling missing values and basic formatting.
  - `data/processed/adult/adult_model_ready.csv` – final, split dataset with engineered features and `split` column (`train`/`test`).
- **Predictions:** Model outputs are saved to `data/predictions/` with filenames prefixed by the model (e.g., `lr_preds.csv`).

> **Note:** The processed CSVs are already included. If you want to rebuild them from the raw Adult dataset, follow the preprocessing notebook (`notebooks/1_Preprocessing.ipynb`).

---

## Repository Structure
```
fairness-awareness-ML/
├── data/                   # Raw data, processed datasets, and saved predictions
├── notebooks/              # Exploration, preprocessing, and experiment notebooks
├── reports/                # Slide decks and project report exports
├── src/
│   ├── main.py             # Step 1: baseline training and fairness evaluation
│   ├── main_step2.py       # Step 2: baseline + Equal Opportunity post-processing
│   ├── metrics/            # Fairness metric implementations
│   ├── models/             # Model training utilities (LR, RF, XGBoost)
│   ├── preprocessing/      # Feature preprocessing pipeline
│   ├── techniques/         # Fairness mitigation methods
│   └── utils/              # Helpers (e.g., saving predictions)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Setup
1. **Python version:** 3.10+ recommended.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data:** Ensure the processed files under `data/processed/adult/` are present (included in the repository). If missing, regenerate them via `notebooks/1_Preprocessing.ipynb`.

---

## Running Experiments
### 1) Baseline models (Step 1)
Trains Logistic Regression, Random Forest, and XGBoost using a shared preprocessing pipeline and reports both performance and fairness metrics.
```bash
python src/main.py
```
**What happens:**
- Loads `adult_model_ready.csv` and constructs train/test splits.
- Removes sensitive attributes from the feature set before fitting.
- Prints accuracy, precision, recall, F1, and fairness metrics (Statistical Parity Difference, Disparate Impact) for `sex` and `race_binary`.
- Saves predictions to `data/predictions/`.

### 2) Equal Opportunity post-processing (Step 2)
Applies a post-processing Equal Opportunity adjustment targeting the `sex` attribute to reduce TPR gaps while retaining model structure.
```bash
python src/main_step2.py
```
**What happens:**
- Trains the same three baseline models.
- Computes TPR gaps between Male and Female groups.
- Lowers the decision threshold for the unprivileged group until TPRs align more closely with the privileged group.
- Records metrics before and after the adjustment for comparison.

### 3) Notebooks
For interactive exploration or rebuilding datasets, open the notebooks in order:
1. `notebooks/1_Preprocessing.ipynb` – data cleaning and train/test split creation.
2. `notebooks/2_metrics_and_models.ipynb` – metric definitions and baseline experiments.
3. `notebooks/3_Train_XGBoost.ipynb` – additional tuning for XGBoost.
4. `notebooks/4_EDA_adult_income.ipynb` – exploratory data analysis.

---

## Fairness Metrics
Implemented in `src/metrics/fairness.py`:
- **Demographic Parity (DP):** Positive prediction rate per group.
- **Statistical Parity Difference (SPD):** Difference in positive rates between privileged and unprivileged groups (0 is ideal).
- **Disparate Impact (DI):** Ratio of positive rates (values near 1 indicate parity).

Equal Opportunity post-processing (in `src/techniques/equal_opportunity.py`) adjusts decision thresholds to close the True Positive Rate gap between groups after model training.

---

## Reproducibility Notes
- Random seeds are not centrally fixed in training scripts; results may vary slightly run to run.
- Preprocessing uses `StandardScaler` for numeric features and one-hot encoding for categorical features while dropping sensitive attributes from the feature matrix.
- Sensitive attributes are only used for fairness evaluation and post-processing.

---

## Contact
For questions or collaboration requests, please open an issue or reach out to the course project team.
