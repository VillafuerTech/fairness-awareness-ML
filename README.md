# Fairness-Aware Candidate Pre-Screening

This project investigates **fairness in an automated hiring pre-screening system** using the Adult (Census Income) dataset. We simulate a large company that receives thousands of online job applications and uses a model as a **first filter** to spot candidates who are likely to have higher earning potential (a proxy for seniority and experience).

In this setup:

- Applicants **predicted to have income `>50K`** are **fast-tracked for human review and interviews**.
- Applicants **predicted `<=50K`** are **deprioritized** and receive fewer or no interview opportunities.

If the model **systematically underestimates** women or non-white candidates, they are invited to interviews **less often**, reinforcing inequality in who advances to better-paid positions and leadership tracks.

This repository was developed for the *Trustworthy Machine Learning 2025 – Task 3 (Fairness-aware ML)* course project.

---

## Problem Setting

A large company wants to quickly identify promising candidates among a very large pool of online applicants. To reduce manual workload, it deploys an ML model as a **pre-screening tool**:

- **Input:** demographic and employment-like features (age, education, occupation, work hours, etc.), similar to census data.
- **Output:** a binary prediction of whether the candidate is likely to have an annual income **above 50K**.
- **Decision rule:**
  - If predicted `>50K` → candidate is **fast-tracked for human review and interviews**.
  - If predicted `<=50K` → candidate is **not prioritized**, and may never be seen by a recruiter.

This is a **high-stakes** decision: being fast-tracked strongly affects who gets access to interviews, job offers, and ultimately higher salaries.

Fairness concerns:

- If the model **underestimates** the likelihood of high income for women or non-white candidates, they receive **fewer interview invitations**.
- Over time, this can **lock underrepresented groups out** of senior, better-paid roles and leadership positions.
- The model may appear accurate overall, yet **unfairly benefit privileged groups** if fairness metrics are not examined.

Our project uses the Adult dataset to simulate this scenario and to study how standard ML models behave when used as a **candidate pre-screening filter**, and how post-processing can mitigate unfair treatment between groups.

---

## Project Goals

- **Measure discrimination** present in the dataset and in standard classifiers when they are used as a hiring pre-screening tool.
- Train baseline models (Logistic Regression, Random Forest, XGBoost) and report both:
  - **Predictive performance metrics**, and
  - **Fairness metrics** across demographic groups.
- Apply **Equal Opportunity post-processing** to reduce disparities between privileged and unprivileged groups **without retraining** the models.
- Provide **transparent, reproducible experiments** to analyze the trade-off between **accuracy** and **group fairness** in a realistic pre-screening pipeline.

---

## Dataset

- **Source:** UCI Adult (Census Income) dataset.  
- **Size:** ~50k individuals with demographic and employment attributes and a binary income label (`>50K` / `<=50K`).  

### Task interpretation in this project

We keep the original binary label, but reinterpret it for hiring:

- `income > 50K` → candidate is treated as **“high-earning / higher-seniority profile”**, and thus **fast-tracked**.
- `income <= 50K` → candidate is treated as **“lower earning / lower-seniority profile”**, and thus **deprioritized**.

### Protected attributes

- **Primary:** `sex` (Male / Female)  
- **Secondary (derived):** `race_binary` (White / Non-White)

These attributes are:

- **Not used as features** for model training.
- Only used for:
  - **Fairness analysis**, and
  - **Post-processing** (group-specific threshold adjustments).

### Processed files

- `data/processed/adult/adult_clean.csv`  
  Cleaned dataset after handling missing values and basic formatting.

- `data/processed/adult/adult_model_ready.csv`  
  Final dataset with engineered features and a `split` column (`train` / `test`).

### Predictions

Model predictions are saved under `data/predictions/` with filenames prefixed by the model name, e.g.:

- `lr_preds.csv` for Logistic Regression  
- `rf_preds.csv` for Random Forest  
- `xgb_preds.csv` for XGBoost  

> **Note:** Processed CSVs are included. To rebuild them from the raw Adult dataset, use `notebooks/1_Preprocessing.ipynb`.

---

## Repository Structure

```text
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
│   └── utils/              # Helper functions (e.g., saving predictions)
├── requirements.txt        # Python dependencies
└── README.md
````

---

## Setup

1. **Python version**

   * Recommended: **Python 3.10+**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Check data availability**

   * Verify that `data/processed/adult/adult_model_ready.csv` exists.
   * If the processed data is missing or you want to regenerate it:

     * Open and run `notebooks/1_Preprocessing.ipynb`.

---

## Running Experiments

### 1) Baseline models (Step 1)

This stage trains Logistic Regression, Random Forest, and XGBoost using a shared preprocessing pipeline, and evaluates how they behave as candidate pre-screening models.

```bash
python src/main.py
```

**What this script does:**

* Loads `adult_model_ready.csv` and builds train/test splits based on the `split` column.
* Applies preprocessing:

  * Standard scaling for numeric features.
  * One-hot encoding for categorical features.
  * Sensitive attributes (`sex`, `race`, `race_binary`) are dropped from the feature matrix used for training.
* Trains baseline models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Computes standard **performance metrics**:

  * Accuracy, Precision, Recall, F1.
* Computes **fairness metrics** for:

  * `sex` (Male vs Female)
  * `race_binary` (White vs Non-White)
* Interprets the positive class `>50K` as **“fast-tracked candidate”** in the hiring pipeline.
* Saves prediction files for each model in `data/predictions/`.

---

### 2) Equal Opportunity post-processing (Step 2)

This stage applies **post-processing** to explicitly target Equal Opportunity in the hiring scenario, focusing on the `sex` attribute.

```bash
python src/main_step2.py
```

**What this script does:**

* Re-trains the same baseline models as in Step 1.
* Calculates True Positive Rates (TPRs) for Male and Female groups, where:

  * True Positive = a genuinely high-earning (`>50K`) candidate correctly predicted as “fast-tracked”.
* Identifies **TPR gaps** between the privileged group (e.g., Male) and the unprivileged group (e.g., Female).
* For the unprivileged group, **lowers the decision threshold** for predicting `>50K` until the TPR moves closer to that of the privileged group.
* Recomputes performance and fairness metrics **before and after** applying this post-processing.
* Allows analysis of the trade-off between:

  * Not “giving up” too much overall accuracy, and
  * Avoiding systematic under-selection of women in the pre-screening stage.

---

### 3) Notebooks

Interactive notebooks for exploration and custom experiments:

1. `notebooks/1_Preprocessing.ipynb`
   Data cleaning, handling missing values, and creating train/test splits.

2. `notebooks/2_metrics_and_models.ipynb`
   Implementation of fairness metrics and baseline experiments.

3. `notebooks/3_Train_XGBoost.ipynb`
   Additional tuning and analysis of XGBoost.

4. `notebooks/4_EDA_adult_income.ipynb`
   Exploratory data analysis focusing on income distribution and group differences.

---

## Fairness Metrics

Implemented in `src/metrics/fairness.py`:

* **Demographic Parity (DP)**

  * Measures the **rate of positive predictions** (fast-tracked candidates) in each group.
  * In this project, “positive” = predicted `>50K` = **fast-tracked for interview**.

* **Statistical Parity Difference (SPD)**

  * Difference in positive prediction rates between privileged and unprivileged groups.
  * Ideal value: **0** (both groups are fast-tracked at the same rate).

* **Disparate Impact (DI)**

  * Ratio of positive prediction rates: (unprivileged group) / (privileged group).
  * Values close to **1** indicate similar odds of being fast-tracked.

### Equal Opportunity post-processing

Implemented in `src/techniques/equal_opportunity.py`:

* Targets **Equal Opportunity**, which in this context means that:

  * Among candidates who truly belong to the high-income group (`>50K`),
  * Women and men should have **similar chances** of being predicted as “fast-tracked”.
* This is enforced (approximately) by:

  * Adjusting the decision threshold *per group* after training,
  * Without retraining or changing model parameters.

---

## Reproducibility Notes

* Random seeds are **not centrally fixed**, so metric values can differ slightly across runs.
* Preprocessing:

  * Uses `StandardScaler` for numeric features.
  * Uses one-hot encoding for categorical variables.
  * Drops sensitive attributes from the features supplied to the models.
* Sensitive attributes (`sex`, `race_binary`) are:

  * Kept in the dataset for **evaluation and analysis**.
  * Used only for computing group metrics and for group-specific post-processing (threshold adjustments).

---

