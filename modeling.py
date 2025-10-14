import os
import sys
import warnings
from dotenv import load_dotenv
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score
)
import mlflow
from mlflow.client import MlflowClient

# Nonaktifkan warning
warnings.filterwarnings("ignore")
load_dotenv()


def run_xgb_model_mlflow(df, dataset_path):
    # === 1️⃣ Setup koneksi ke DagsHub MLflow ===
    uri_dagshub = "https://dagshub.com/NauraaSalsabila/DSP_ATTRITION.mlflow"
    mlflow.set_tracking_uri(uri_dagshub)
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    # === 2️⃣ Setup MLflow Experiment ===
    experiment_name = "attrition_prediction"
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(name=experiment_name)
        print(f"⚙️ Eksperimen baru dibuat: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"ℹ️ Eksperimen aktif: '{experiment_name}' (ID: {experiment_id})")

    # === 3️⃣ Siapkan data untuk training ===
    if "Attrition" not in df.columns:
        raise ValueError("❌ Kolom 'Attrition' tidak ditemukan di dataset!")

    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])
    print(f"✅ Dataset siap: {df.shape[0]} baris, {df.shape[1]} kolom.")

    # === 4️⃣ Split data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === 5️⃣ Jalankan MLflow Autolog untuk XGBoost ===
    mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="xgb-default-model", experiment_id=experiment_id):
        # === 6️⃣ Log dataset ke MLflow ===
        try:
            from mlflow.data import from_pandas
            dataset_info = from_pandas(df, name="preprocessed_data")
            mlflow.log_input(dataset_info)
            print("📦 Dataset berhasil diregistrasi ke MLflow (tab Dataset).")
        except Exception as e:
            print(f"⚠️ Gagal log dataset: {e}")

        # === 7️⃣ Inisialisasi dan training model XGBoost ===
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
        )

        # Logging parameter manual untuk keamanan
        for param, val in model.get_params().items():
            mlflow.log_param(param, val)

        model.fit(X_train, y_train)

        # === 8️⃣ Prediksi dan Evaluasi ===
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        # ---- TRAIN METRICS ----
        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train)
        train_rec = recall_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train)
        train_logloss = log_loss(y_train, y_prob_train)
        train_roc = roc_auc_score(y_train, y_prob_train)

        # ---- TEST METRICS ----
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test)
        test_rec = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)

        # === 9️⃣ Logging metrics manual ke MLflow ===
        metrics = {
            "training_accuracy_score": train_acc,
            "training_precision_score": train_prec,
            "training_recall_score": train_rec,
            "training_f1_score": train_f1,
            "training_log_loss": train_logloss,
            "training_roc_auc": train_roc,
            "training_score": train_acc,
            "test accuracy": test_acc,
            "test precision": test_prec,
            "test recall": test_rec,
            "test f1-score": test_f1,
        }

        for key, val in metrics.items():
            mlflow.log_metric(key, float(val))

        # === 10️⃣ Cetak hasil di terminal ===
        print("\n📊 Hasil Evaluasi (XGBoost - Training):")
        print(f"   Accuracy : {train_acc:.4f}")
        print(f"   Precision: {train_prec:.4f}")
        print(f"   Recall   : {train_rec:.4f}")
        print(f"   F1 Score : {train_f1:.4f}")
        print(f"   Log Loss : {train_logloss:.4f}")
        print(f"   ROC AUC  : {train_roc:.4f}")

        print("\n📊 Hasil Evaluasi (XGBoost - Testing):")
        print(f"   Accuracy : {test_acc:.4f}")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   Recall   : {test_rec:.4f}")
        print(f"   F1 Score : {test_f1:.4f}")

        print("\n✅ Model berhasil dilatih dan di-log ke DagsHub MLflow.")
        print("🔗 Cek hasil training di dashboard DagsHub kamu.")


if __name__ == "__main__":
    dataset_path = "data/preprocessed_data.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ File dataset '{dataset_path}' tidak ditemukan.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    run_xgb_model_mlflow(df, dataset_path)
