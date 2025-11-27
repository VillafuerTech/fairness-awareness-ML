# Fairness-aware Income Prediction

Course project for **Trustworthy Machine Learning 2025 – Task 3 (Fairness-aware ML)**.

This repository contains our case study on **fairness in automated income prediction** using the Adult Census Income dataset.

---

## 1. Project Overview

- **Goal:** Study how standard ML models can exhibit discrimination and how fairness-aware methods can mitigate it.
- **Application story:**  
  Predict whether a person earns **> 50k USD/year** for a hypothetical use case such as:
  - Prioritizing customers for **financial services**, or
  - Prioritizing individuals for **tax audits**.
- **Main questions:**
  - How much discrimination exists in the **data** and in **baseline models**?
  - Can a fairness-aware model **reduce discrimination** while keeping reasonable predictive performance?

---

## 2. Dataset

- **Name:** Adult (Census Income) dataset
- **Source:** UCI Machine Learning Repository
- **Size:** ~50 000 records, with demographic and income information.
- **Target variable (`y`):**
  - `1` → income `>50K`
  - `0` → income `<=50K`
- **Protected attribute (`A`):**
  - Primary: **sex** (male / female)
  - Optional secondary: **race**

Data files will be stored in `data/`.

---

## 3. Repository Structure

```text
fairness-project/
├── data/                # Raw and processed datasets
├── notebooks/           # EDA, baselines, fairness-aware experiments
├── src/                 # Reusable code (data loading, models, metrics)
├── reports/
│   ├── slides_step1/    # Presentation I (Dec 1)
│   ├── slides_step2/    # Presentation II (Dec 8)
│   └── final_report/    # Final report (Dec 15)
└── README.md

```
