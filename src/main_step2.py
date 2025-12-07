import pandas as pd

from src.preprocessing.preprocess import build_preprocessing_pipeline
from src.models.train_lr import train_lr
from src.models.train_rf import train_rf
from src.models.train_xgb import train_xgb
from src.metrics.fairness import compute_fairness_metrics
from src.utils.utils import save_predictions
from src.techniques.equal_opportunity import equal_opportunity_postprocessing

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================================
# GLOBAL METRIC STORE
# =====================================================================

all_metrics = []


# =====================================================================
# HELPERS
# =====================================================================

def record_metrics(model_name, condition, y_true, y_pred, df_test, tpr_gap=None):
    fair = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["sex"],
        privileged_group="Male",
    )

    all_metrics.append({
        "Model": model_name,
        "Condition": condition,
        "Accuracy": accuracy_score(y_true, y_pred),
        "SPD": fair["SPD"],
        "DI": fair["DI"],
        "TPR_gap": tpr_gap
    })


def print_performance(model_name, y_true, y_pred):
    print(f"\n {model_name} PERFORMANCE ")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))


def print_fairness(model_name, y_true, y_pred, df_test):
    print(f"\n {model_name} FAIRNESS ")

    fairness_sex = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["sex"],
        privileged_group="Male",
    )
    print("Sex fairness :", fairness_sex)

    fairness_race = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["race_binary"],
        privileged_group="White",
    )
    print("Race fairness:", fairness_race)


def compute_tpr_by_sex(y_true, y_pred, df_test):
    y_true = y_true.to_numpy()
    y_pred = y_pred

    def tpr(actual, pred):
        mask = (actual == 1)
        if mask.sum() == 0:
            return 0.0
        return (pred[mask] == 1).mean()

    male_mask = (df_test["sex"] == "Male").to_numpy()
    female_mask = (df_test["sex"] == "Female").to_numpy()

    tpr_male = tpr(y_true[male_mask], y_pred[male_mask])
    tpr_female = tpr(y_true[female_mask], y_pred[female_mask])
    gap = tpr_male - tpr_female

    return tpr_male, tpr_female, gap


def print_tpr(label, tpr_male, tpr_female, gap):
    print(f"\n {label} TPR BY SEX ")
    print(f"TPR (Male)   : {tpr_male:.4f}")
    print(f"TPR (Female) : {tpr_female:.4f}")
    print(f"TPR gap (M-F): {gap:.4f}")


# =====================================================================
# MAIN
# =====================================================================

def main():

    model_ready_path = "data/processed/adult/adult_model_ready.csv"
    df = pd.read_csv(model_ready_path)

    df["race_binary"] = df["race"].apply(lambda r: "White" if r == "White" else "Non-White")

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    y_train = (df_train["income"] == ">50K").astype(int)
    y_test = (df_test["income"] == ">50K").astype(int)

    X_train = df_train.drop(columns=["income", "sex", "race", "race_binary", "split"])
    X_test = df_test.drop(columns=["income", "sex", "race", "race_binary", "split"])

    preprocess, _, _ = build_preprocessing_pipeline(df_train)

    # -----------------------------------------------------------------
    # RAW FAIRNESS (no model)
    # -----------------------------------------------------------------
    print("\n FAIRNESS BEFORE MODEL TRAINING (RAW LABELS) ")

    raw_fair_sex = compute_fairness_metrics(
        y_true=y_test, y_pred=y_test,
        sensitive=df_test["sex"], privileged_group="Male"
    )
    print("Raw Fairness (SEX):", raw_fair_sex)

    raw_fair_race = compute_fairness_metrics(
        y_true=y_test, y_pred=y_test,
        sensitive=df_test["race_binary"], privileged_group="White"
    )
    print("Raw Fairness (RACE):", raw_fair_race)

    # -----------------------------------------------------------------
    # TRAIN MODELS (BASELINE)
    # -----------------------------------------------------------------

    # ---------------- Logistic Regression ----------------
    print("\nTraining Logistic Regression...")
    lr_clf = train_lr(preprocess, X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    y_proba_lr = lr_clf.predict_proba(X_test)[:, 1]

    save_predictions("lr_step2_base", y_test, y_pred_lr, X_test.index)
    print_performance("Logistic Regression (BASE)", y_test, y_pred_lr)
    print_fairness("Logistic Regression (BASE)", y_test, y_pred_lr, df_test)

    tpr_m_lr, tpr_f_lr, gap_lr = compute_tpr_by_sex(y_test, y_pred_lr, df_test)
    print_tpr("Logistic Regression (BASE)", tpr_m_lr, tpr_f_lr, gap_lr)

    record_metrics("LR", "Before EO", y_test, y_pred_lr, df_test, gap_lr)

    # ---------------- Random Forest ----------------
    print("\nTraining Random Forest...")
    rf_clf = train_rf(preprocess, X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

    save_predictions("rf_step2_base", y_test, y_pred_rf, X_test.index)
    print_performance("Random Forest (BASE)", y_test, y_pred_rf)
    print_fairness("Random Forest (BASE)", y_test, y_pred_rf, df_test)

    tpr_m_rf, tpr_f_rf, gap_rf = compute_tpr_by_sex(y_test, y_pred_rf, df_test)
    print_tpr("Random Forest (BASE)", tpr_m_rf, tpr_f_rf, gap_rf)

    record_metrics("RF", "Before EO", y_test, y_pred_rf, df_test, gap_rf)

    # ---------------- XGBoost ----------------
    print("\nTraining XGBoost...")
    xgb_clf = train_xgb(preprocess, X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    save_predictions("xgb_step2_base", y_test, y_pred_xgb, X_test.index)
    print_performance("XGBoost (BASE)", y_test, y_pred_xgb)
    print_fairness("XGBoost (BASE)", y_test, y_pred_xgb, df_test)

    tpr_m_xgb, tpr_f_xgb, gap_xgb = compute_tpr_by_sex(y_test, y_pred_xgb, df_test)
    print_tpr("XGBoost (BASE)", tpr_m_xgb, tpr_f_xgb, gap_xgb)

    record_metrics("XGB", "Before EO", y_test, y_pred_xgb, df_test, gap_xgb)

    # -----------------------------------------------------------------
    # APPLY EQUAL OPPORTUNITY (SEX)
    # -----------------------------------------------------------------

    print("\n===== APPLYING EQUAL OPPORTUNITY (SEX) TO ALL MODELS =====")

    # ---------------- Logistic Regression + EO ----------------
    y_lr_eo, info_lr = equal_opportunity_postprocessing(
        y_true=y_test.values,
        y_pred_proba=y_proba_lr,
        sensitive_attr=df_test["sex"].values,
    )
    save_predictions("lr_step2_eo", y_test, y_lr_eo, X_test.index)

    print("\n--- Logistic Regression + EO ---")
    print("EO info:", info_lr)
    print_performance("Logistic Regression (EO)", y_test, y_lr_eo)
    print_fairness("Logistic Regression (EO)", y_test, y_lr_eo, df_test)

    tpr_m_lr_eo, tpr_f_lr_eo, gap_lr_eo = compute_tpr_by_sex(y_test, y_lr_eo, df_test)
    print_tpr("Logistic Regression (EO)", tpr_m_lr_eo, tpr_f_lr_eo, gap_lr_eo)

    record_metrics("LR", "After EO", y_test, y_lr_eo, df_test, gap_lr_eo)

    # ---------------- Random Forest + EO ----------------
    y_rf_eo, info_rf = equal_opportunity_postprocessing(
        y_true=y_test.values,
        y_pred_proba=y_proba_rf,
        sensitive_attr=df_test["sex"].values,
    )
    save_predictions("rf_step2_eo", y_test, y_rf_eo, X_test.index)

    print("\n--- Random Forest + EO ---")
    print("EO info:", info_rf)
    print_performance("Random Forest (EO)", y_test, y_rf_eo)
    print_fairness("Random Forest (EO)", y_test, y_rf_eo, df_test)

    tpr_m_rf_eo, tpr_f_rf_eo, gap_rf_eo = compute_tpr_by_sex(y_test, y_rf_eo, df_test)
    print_tpr("Random Forest (EO)", tpr_m_rf_eo, tpr_f_rf_eo, gap_rf_eo)

    record_metrics("RF", "After EO", y_test, y_rf_eo, df_test, gap_rf_eo)

    # ---------------- XGBoost + EO ----------------
    y_xgb_eo, info_xgb = equal_opportunity_postprocessing(
        y_true=y_test.values,
        y_pred_proba=y_proba_xgb,
        sensitive_attr=df_test["sex"].values,
    )
    save_predictions("xgb_step2_eo", y_test, y_xgb_eo, X_test.index)

    print("\n--- XGBoost + EO ---")
    print("EO info:", info_xgb)
    print_performance("XGBoost (EO)", y_test, y_xgb_eo)
    print_fairness("XGBoost (EO)", y_test, y_xgb_eo, df_test)

    tpr_m_xgb_eo, tpr_f_xgb_eo, gap_xgb_eo = compute_tpr_by_sex(y_test, y_xgb_eo, df_test)
    print_tpr("XGBoost (EO)", tpr_m_xgb_eo, tpr_f_xgb_eo, gap_xgb_eo)

    record_metrics("XGB", "After EO", y_test, y_xgb_eo, df_test, gap_xgb_eo)

    # -----------------------------------------------------------------
    # SAVE METRICS TO CSV
    # -----------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("data/metrics/step2_metrics.csv", index=False)

    print("\n=== STORED ALL METRICS IN data/metrics/step2_metrics.csv ===")
    print(metrics_df)

    print("\n STEP 2 (Equal Opportunity technique for SEX) COMPLETED ")


if __name__ == "__main__":
    main()