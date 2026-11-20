import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

params = yaml.safe_load(open("params.yaml"))["train"]
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

mlflow.set_experiment("wine-demo")
with mlflow.start_run():
    model = RandomForestClassifier(**params, random_state=42).fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="macro"),
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)

joblib.dump(model, "models/model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)