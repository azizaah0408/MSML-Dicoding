import pandas as pd
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Konfigurasi ---
TRAIN_DATA = 'loan_data_train_processed.csv'
TEST_DATA = 'loan_data_test_processed.csv'
TARGET_COL = 'Loan_Status'
MODEL_PATH = 'best_model.pkl'

def load_data():
    """Memuat data training dan testing"""
    print("[INFO] Loading data...")
    if not os.path.exists(TRAIN_DATA) or not os.path.exists(TEST_DATA):
        print(f"❌ Error: File {TRAIN_DATA} atau {TEST_DATA} tidak ditemukan.")
        return None, None, None, None
    
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    
    return X_train, y_train, X_test, y_test

def train_eval_model():
    X_train, y_train, X_test, y_test = load_data()
    
    if X_train is None:
        return

    # Kita coba 2 Algoritma berbeda (Syarat Skilled)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_acc = 0
    best_name = ""
    
    # Loop untuk training dan evaluasi
    for name, model in models.items():
        print(f"\n--- Training Model: {name} ---")
        model.fit(X_train, y_train)
        
        # Prediksi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Akurasi {name}: {acc:.4f}")
        print(classification_report(y_test, preds))
        
        # Cek apakah ini model terbaik?
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # Simpan Model Terbaik
    if best_model:
        print(f"\n[RESULT] Pemenangnya adalah: {best_name} (Akurasi: {best_acc:.4f})")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"[SUCCESS] Model disimpan sebagai '{MODEL_PATH}'")

if __name__ == "__main__":
    train_eval_model()