from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception.exception import CustomException
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from sklearn.model_selection import GridSearchCV

class DetectionModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CustomException(e,sys)

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
            
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for name, model in models.items():
            gs = GridSearchCV(
                model,
                param[name],
                cv=3,
                n_jobs=-1,
                verbose=0,
                scoring="f1",
            )
            gs.fit(X_train, y_train)

            # best_estimator_ is already refitted on full X_train by GridSearchCV
            best = gs.best_estimator_
            models[name] = best            # update in-place so caller gets tuned model

            y_test_pred      = best.predict(X_test)
            test_model_score = f1_score(y_test, y_test_pred)
            report[name]     = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
