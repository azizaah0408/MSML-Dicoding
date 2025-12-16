import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 🔐 KUNCI RAHASIA (Setting Manual biar Gak Usah Login Browser) ---
# Data ini diambil dari info yang kamu kirim tadi
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/azizaah0408/MSML-Dicoding.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "azizaah0408"

# 👇👇👇 PASTE TOKEN (PASSWORD) DARI DAGSHUB DI SINI (Di dalam tanda kutip)
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bb464161a6111e1127426e6c3b1c94f403ac6fb6" 

# --- Konfigurasi Data ---
TRAIN_DATA = 'loan_data_train_processed.csv'
TEST_DATA = 'loan_data_test_processed.csv'
TARGET_COL = 'Loan_Status'
OUTPUT_MODEL = 'best_model_tuned.pkl'

def load_data():
    print("[INFO] Loading data...")
    if not os.path.exists(TRAIN_DATA):
        print("❌ Error: File data tidak ditemukan.")
        return None, None, None, None
    
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    plt.close()
    return "confusion_matrix.png"

def save_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        return "feature_importance.png"
    return None

def run_tuning():
    # Set URI secara manual (Bypass login browser)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None: return

    # Nama Eksperimen di DagsHub
    mlflow.set_experiment("Eksperimen_Loan_Prediction_Advanced")

    with mlflow.start_run():
        print("[INFO] Memulai Tuning ke DagsHub (Mode Token Manual)...")
        
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }

        # Proses Training
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"✅ Akurasi: {acc:.4f}")
        
        # Log ke DagsHub
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)

        # Log Gambar
        cm_file = save_confusion_matrix(y_test, preds)
        mlflow.log_artifact(cm_file)
        
        feat_imp_file = save_feature_importance(best_model, X_train.columns)
        if feat_imp_file:
            mlflow.log_artifact(feat_imp_file)

        # Log Model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Simpan Lokal
        with open(OUTPUT_MODEL, 'wb') as f:
            pickle.dump(best_model, f)
            
        print("[SUCCESS] Selesai! Cek Dashboard DagsHub kamu sekarang.")

if __name__ == "__main__":
    run_tuning()