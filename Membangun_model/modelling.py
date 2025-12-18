import mlflow
import mlflow.sklearn

def train_eval_model():
    X_train, y_train, X_test, y_test = load_data()
    
    if X_train is None: return

    mlflow.set_tracking_uri("http://127.0.0.1:5000") # Arahkan ke Localhost
    mlflow.set_experiment("Latihan Credit Scoring") # Nama eksperimen bebas
    
    # Bungkus training dalam start_run()
    with mlflow.start_run():
        # Pilih model terbaik untuk versi basic ini
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Akurasi: {acc:.4f}")
        
        # LOGGING KE MLFLOW
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan lokal
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    train_eval_model()