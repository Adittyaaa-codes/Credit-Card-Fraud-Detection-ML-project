import os
import sys
from urllib.parse import urlparse

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.utils import save_object, load_object, load_numpy_array
from src.utils.ml_utils import DetectionModel, get_classification_score, evaluate_models

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

import mlflow
import dagshub

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def track_mlflow(self, best_model, classificationmetric):
        dagshub.init(repo_owner='Adittyaaa-codes', repo_name='Credit-Card-FraudDetection-ML-project', mlflow=True)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)
            model_name = type(best_model).__name__
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, name="model", registered_model_name=model_name)
            else:
                mlflow.sklearn.log_model(best_model, name="model")

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.80, random_state=42)
            sample_idx, _ = next(sss.split(X_train, y_train))
            X_fit, y_fit = X_train[sample_idx], y_train[sample_idx]
            logging.info(f"Training on 20% sample: {len(X_fit):,} rows out of {len(X_train):,}")

            models = {
                "Logistic Regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))
                ]),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
            }

            params = {
                "Logistic Regression": {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
                "Decision Tree": {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 10, 20]},
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
            }

            model_report: dict = evaluate_models(
                X_train=X_fit, y_train=y_fit, X_test=x_test, y_test=y_test,
                models=models, param=params,
            )

            best_model_score = max(model_report.values())
            best_model_name  = max(model_report, key=model_report.get)
            best_model       = models[best_model_name]
            logging.info(f"Best model: '{best_model_name}' | test F1 = {best_model_score:.4f}")

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, classification_test_metric)

            preprocessor = load_object(file_path=self.data_transformation_artifact.preprocessor_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Detection_Model = DetectionModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Detection_Model)
            
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)
            save_object("final_model/preprocessor.pkl", preprocessor)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
 
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

         
            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise CustomException(e,sys)