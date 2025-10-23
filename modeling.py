import os
import sys
import warnings
from dotenv import load_dotenv
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, roc_auc_score
)
import mlflow
from mlflow.client import MlflowClient

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

    # === 3️⃣ Fitur penting berdasarkan hasil analisis feature importance ===
    encoded_features = [
        "JobRole_Human Resources", "JobRole_Healthcare Representative", "JobRole_Research Scientist",
        "JobRole_Sales Executive", "JobRole_Manager", "JobRole_Laboratory Technician",
        "JobRole_Research Director", "JobRole_Manufacturing Director", "JobRole_Sales Representative",
        "OverTime_Yes", "StockOptionLevel", "MaritalStatus_Single", "MaritalStatus_Married",
        "MaritalStatus_Divorced", "EducationField_Human Resources", "EducationField_Life Sciences",
        "EducationField_Marketing", "EducationField_Medical", "EducationField_Other",
        "EducationField_Technical Degree", "JobLevel", "BusinessTravel_Non-Travel",
        "BusinessTravel_Travel_Rarely", "BusinessTravel_Travel_Frequently",
        "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction"
    ]

    # Filter fitur yang tersedia di dataset
    available_features = [f for f in encoded_features if f in df.columns]
    missing = [f for f in encoded_features if f not in df.columns]

    if missing:
        print(f"⚠️ Beberapa fitur tidak ditemukan di dataset dan akan di-skip: {missing}")

    # Ambil fitur yang tersedia + target
    X = df[available_features]
    y = df["Attrition"]

    print(f"✅ Dataset siap: {df.shape[0]} baris | Gunakan {len(available_features)} fitur utama (encoded).")
    print("🔹 Fitur digunakan:", ", ".join(available_features))

    # === 4️⃣ Split data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === 5️⃣ Jalankan MLflow Autolog ===
    mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="xgb-10-feature-model-encoded", experiment_id=experiment_id):
        # === 6️⃣ Log dataset ke MLflow ===
        try:
            from mlflow.data import from_pandas
            dataset_info = from_pandas(df, name="preprocessed_data_encoded")
            mlflow.log_input(dataset_info)
            print("📦 Dataset berhasil diregistrasi ke MLflow.")
        except Exception as e:
            print(f"⚠️ Gagal log dataset: {e}")

        # === 7️⃣ Inisialisasi dan training model ===
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
        )

        # Logging parameter
        for param, val in model.get_params().items():
            mlflow.log_param(param, val)

        model.fit(X_train, y_train)

        # === 8️⃣ Evaluasi ===
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        metrics = {
            "train_acc": accuracy_score(y_train, y_pred_train),
            "train_prec": precision_score(y_train, y_pred_train),
            "train_rec": recall_score(y_train, y_pred_train),
            "train_f1": f1_score(y_train, y_pred_train),
            "train_logloss": log_loss(y_train, y_prob_train),
            "train_roc": roc_auc_score(y_train, y_prob_train),
            "test_acc": accuracy_score(y_test, y_pred_test),
            "test_prec": precision_score(y_test, y_pred_test),
            "test_rec": recall_score(y_test, y_pred_test),
            "test_f1": f1_score(y_test, y_pred_test),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        print("\n📊 Hasil Evaluasi (Training/Test):")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")

        # === 9️⃣ Register ke MLflow Model Registry ===
        try:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="xgb-attrition-model-encoded"
            )
            print(f"✅ Model diregister: {registered_model.name} (versi {registered_model.version})")
        except Exception as e:
            print(f"⚠️ Gagal register model: {e}")


if __name__ == "__main__":
    dataset_path = "data/preprocessed_data.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ File dataset '{dataset_path}' tidak ditemukan.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    run_xgb_model_mlflow(df, dataset_path)
