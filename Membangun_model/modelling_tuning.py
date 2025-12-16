import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# --- Konfigurasi ---
TRAIN_DATA = 'loan_data_train_processed.csv'
TEST_DATA = 'loan_data_test_processed.csv'
TARGET_COL = 'Loan_Status'
OUTPUT_MODEL = 'best_model_tuned.pkl'

def load_data():
    print("[INFO] Loading data untuk Tuning...")
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

def run_tuning():
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None: return

    # Definisi Model Dasar
    rf = RandomForestClassifier(random_state=42)

    # Definisi Parameter yang akan dicoba-coba (Eksperimen)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    print("\n[INFO] Memulai Hyperparameter Tuning (GridSearch)...")
    print("Sedang mencari settingan terbaik, mohon tunggu sebentar...")

    # Mencari kombinasi terbaik
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Hasil Terbaik
    best_model = grid_search.best_estimator_
    print(f"\n✅ Settingan Terbaik: {grid_search.best_params_}")

    # Evaluasi
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Akurasi Model Tuning: {acc:.4f}")

    # Simpan
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"[SUCCESS] Model Tuning disimpan ke '{OUTPUT_MODEL}'")

if __name__ == "__main__":
    run_tuning()