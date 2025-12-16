import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Konfigurasi
DATA_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
TARGET_COLUMN = "Loan_Status"
DROP_COLUMNS = ["Loan_ID"]

def load_data(url):
    """Fungsi untuk memuat data dari URL"""
    print(f"[INFO] Memuat data dari: {url}...")
    try:
        df = pd.read_csv(url)
        print(f"[SUCCESS] Data berhasil dimuat! Dimensi: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal memuat data: {e}")
        return None

def get_preprocessing_pipeline():
    """Fungsi yang mendefinisikan langkah pembersihan data (Blueprint)"""
    
    # Kolom Numerik (Angka)
    numerical_cols = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History'
    ]

    # Kolom Kategorikal (Teks)
    categorical_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'Property_Area'
    ]

    # Pipeline Numerik: Isi Kosong (Median) -> Standarisasi
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline Kategorikal: Isi Kosong (Modus) -> OneHot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Gabungkan
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor, numerical_cols, categorical_cols

def process_and_save(df):
    """Fungsi utama untuk menjalankan preprocessing dan menyimpan hasil"""
    print("[INFO] Memulai Preprocessing...")
    
    # 1. Pisahkan Fitur dan Target
    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)
    y = df[TARGET_COLUMN]

    # 2. Encode Target (Y/N -> 1/0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"[INFO] Target Encoded: {le.classes_} -> [0, 1]")

    # 3. Ambil Pipeline
    preprocessor, num_cols, cat_cols = get_preprocessing_pipeline()

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Fit & Transform
    print("[INFO] Sedang melakukan cleaning & scaling...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 6. Rapikan Nama Kolom (Opsional, biar file CSV ada headernya)
    # Kita perlu fit onehot dulu ke data untuk dapet nama kolom baru
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(ohe_feature_names)

    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # 7. Simpan ke File
    # Gabungkan fitur + target
    train_set = pd.concat([X_train_df, pd.Series(y_train, name=TARGET_COLUMN)], axis=1)
    test_set = pd.concat([X_test_df, pd.Series(y_test, name=TARGET_COLUMN)], axis=1)

    # Pastikan folder output ada (opsional, simpan di root folder project)
    output_dir = ".." # Simpan di folder atasnya (root) biar rapi
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_path = os.path.join(output_dir, "loan_data_train_processed.csv")
    test_path = os.path.join(output_dir, "loan_data_test_processed.csv")

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)

    print(f"[SUCCESS] Data Train disimpan di: {train_path}")
    print(f"[SUCCESS] Data Test disimpan di: {test_path}")

if __name__ == "__main__":
    # Langkah Eksekusi
    df = load_data(DATA_URL)
    if df is not None:
        process_and_save(df)