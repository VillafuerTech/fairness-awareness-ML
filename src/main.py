import pandas as pd
from pathlib import Path

from src.preprocessing.preprocess import build_preprocessing_pipeline
from src.models.train_lr import train_lr
from src.models.train_rf import train_rf
from src.models.train_xgb import train_xgb
from src.metrics.fairness import compute_fairness_metrics
from src.utils.utils import save_predictions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        privileged_group="Male"
    )
    print("Sex fairness :", fairness_sex)

    fairness_race = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=df_test["race_binary"],
        privileged_group="White"
    )
    print("Race fairness:", fairness_race)


def main():

    model_ready_path = "data/processed/adult/adult_model_ready.csv"
    df = pd.read_csv(model_ready_path)

    # Create binary race column
    df["race_binary"] = df["race"].apply(lambda r: "White" if r == "White" else "Non-White")

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    y_train = (df_train["income"] == ">50K").astype(int)
    y_test  = (df_test["income"] == ">50K").astype(int)

    X_train = df_train.drop(columns=["income", "sex", "race", "race_binary", "split"])
    X_test  = df_test.drop(columns=["income", "sex", "race", "race_binary", "split"])

    preprocess, num_cols, cat_cols = build_preprocessing_pipeline(df_train)

    
    #  Fairness BEFORE model training

    print("\n FAIRNESS BEFORE MODEL TRAINING ")

    raw_fair_sex = compute_fairness_metrics(
        y_true=y_test,
        y_pred=y_test,
        sensitive=df_test["sex"],
        privileged_group="Male"
    )
    print("Raw Fairness (SEX):", raw_fair_sex)

    raw_fair_race = compute_fairness_metrics(
        y_true=y_test,
        y_pred=y_test,
        sensitive=df_test["race_binary"],
        privileged_group="White"
    )
    print("Raw Fairness (RACE):", raw_fair_race)


    print("\nTraining Logistic Regression...")
    lr_clf = train_lr(preprocess, X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)

    save_predictions("lr", y_test, y_pred_lr, X_test.index)

    print_performance("Logistic Regression", y_test, y_pred_lr)
    print_fairness("Logistic Regression", y_test, y_pred_lr, df_test)


    print("\nTraining Random Forest...")
    rf_clf = train_rf(preprocess, X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    save_predictions("rf", y_test, y_pred_rf, X_test.index)

    print_performance("Random Forest", y_test, y_pred_rf)
    print_fairness("Random Forest", y_test, y_pred_rf, df_test)

    print("\nTraining XGBoost...")
    xgb_clf = train_xgb(preprocess, X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)

    save_predictions("xgb", y_test, y_pred_xgb, X_test.index)

    print_performance("XGBoost", y_test, y_pred_xgb)
    print_fairness("XGBoost", y_test, y_pred_xgb, df_test)

    print("\n STEP 1 COMPLETED ")

if __name__ == "__main__":
    main()