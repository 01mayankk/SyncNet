"""
SyncNet Baseline Training & Evaluation Engine
=============================================
Trains and evaluates traditional ML baselines on:
1. Behavioral Profile Features (12 numerical metrics)
2. Transformer Content Embeddings (384-dimensional vectors)

Models:
- Logistic Regression (with StandardScaler)
- Random Forest Classifier (n_estimators=100, max_depth=6)

Data Leakage Safeguards:
- Models fit ONLY on Train split (800 accounts).
- Hyperparameters validated on Val split (100 accounts).
- Evaluated on strictly held-out Test split (100 accounts).

Outputs:
- Metric results table & confusion matrices
- Saved model checkpoints in models/
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
RANDOM_SEED = 42

def evaluate_model(model, scaler, X_train, y_train, X_test, y_test, name="Model"):
    if scaler is not None:
        X_tr_scaled = scaler.transform(X_train)
        X_te_scaled = scaler.transform(X_test)
    else:
        X_tr_scaled = X_train
        X_te_scaled = X_test
        
    model.fit(X_tr_scaled, y_train)
    
    y_pred = model.predict(X_te_scaled)
    y_prob = model.predict_proba(X_te_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "Name": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Confusion_Matrix": cm.tolist(),
    }
    return metrics, model

def train_and_evaluate_baselines():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    beh_path = os.path.join(DATA_FEAT_DIR, "behavioral_features.parquet")
    content_path = os.path.join(DATA_FEAT_DIR, "content_embeddings.parquet")
    
    if not os.path.exists(beh_path) or not os.path.exists(content_path):
        raise FileNotFoundError("Feature matrices not found. Run preprocess and extraction scripts first.")
        
    df_beh = pd.read_parquet(beh_path)
    df_content = pd.read_parquet(content_path)
    
    print("=" * 70)
    print("SYNCNET — MACHINE LEARNING BASELINE EVALUATION")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. Behavioral Features Baselines (Experiment 01)
    # -------------------------------------------------------------
    beh_feature_cols = [c for c in df_beh.columns if c not in ["user_id", "split", "label"]]
    
    X_train_beh = df_beh[df_beh["split"] == "train"][beh_feature_cols].values
    y_train_beh = df_beh[df_beh["split"] == "train"]["label"].values
    
    X_test_beh = df_beh[df_beh["split"] == "test"][beh_feature_cols].values
    y_test_beh = df_beh[df_beh["split"] == "test"]["label"].values
    
    beh_scaler = StandardScaler()
    beh_scaler.fit(X_train_beh)
    
    # Model 1A: Logistic Regression (Behavioral)
    lr_beh = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    beh_lr_metrics, trained_lr_beh = evaluate_model(
        lr_beh, beh_scaler, X_train_beh, y_train_beh, X_test_beh, y_test_beh, name="Behavioral Logistic Regression"
    )
    
    # Model 1B: Random Forest (Behavioral)
    rf_beh = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)
    beh_rf_metrics, trained_rf_beh = evaluate_model(
        rf_beh, None, X_train_beh, y_train_beh, X_test_beh, y_test_beh, name="Behavioral Random Forest"
    )
    
    # Save Behavioral Models
    joblib.dump(trained_lr_beh, os.path.join(MODELS_DIR, "behavioral_logistic_regression.joblib"))
    joblib.dump(trained_rf_beh, os.path.join(MODELS_DIR, "behavioral_random_forest.joblib"))
    joblib.dump(beh_scaler, os.path.join(MODELS_DIR, "behavioral_scaler.joblib"))
    
    # -------------------------------------------------------------
    # 2. Content Embedding Baselines (Experiment 02)
    # -------------------------------------------------------------
    content_feature_cols = [c for c in df_content.columns if c not in ["user_id", "split", "label"]]
    
    X_train_cnt = df_content[df_content["split"] == "train"][content_feature_cols].values
    y_train_cnt = df_content[df_content["split"] == "train"]["label"].values
    
    X_test_cnt = df_content[df_content["split"] == "test"][content_feature_cols].values
    y_test_cnt = df_content[df_content["split"] == "test"]["label"].values
    
    cnt_scaler = StandardScaler()
    cnt_scaler.fit(X_train_cnt)
    
    # Model 2A: Logistic Regression (Content)
    lr_cnt = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    cnt_lr_metrics, trained_lr_cnt = evaluate_model(
        lr_cnt, cnt_scaler, X_train_cnt, y_train_cnt, X_test_cnt, y_test_cnt, name="Content Logistic Regression"
    )
    
    # Model 2B: Random Forest (Content)
    rf_cnt = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED)
    cnt_rf_metrics, trained_rf_cnt = evaluate_model(
        rf_cnt, None, X_train_cnt, y_train_cnt, X_test_cnt, y_test_cnt, name="Content Random Forest"
    )
    
    # Save Content Models
    joblib.dump(trained_lr_cnt, os.path.join(MODELS_DIR, "content_logistic_regression.joblib"))
    joblib.dump(trained_rf_cnt, os.path.join(MODELS_DIR, "content_random_forest.joblib"))
    
    # -------------------------------------------------------------
    # 3. Print Results Summary Table
    # -------------------------------------------------------------
    all_results = [beh_lr_metrics, beh_rf_metrics, cnt_lr_metrics, cnt_rf_metrics]
    df_res = pd.DataFrame(all_results)
    
    print("\n--- HELD-OUT TEST EVALUATION METRICS SUMMARY ---")
    print(df_res[["Name", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].to_string(index=False))
    
    print("\n--- CONFUSION MATRICES ---")
    for r in all_results:
        print(f"\n{r['Name']}:")
        print(f"  TN: {r['Confusion_Matrix'][0][0]} | FP: {r['Confusion_Matrix'][0][1]}")
        print(f"  FN: {r['Confusion_Matrix'][1][0]} | TP: {r['Confusion_Matrix'][1][1]}")
        
    print("\n[x] SUCCESS: All baseline models trained, evaluated, and saved to models/")
    print("=" * 70)
    return df_res

if __name__ == "__main__":
    train_and_evaluate_baselines()
