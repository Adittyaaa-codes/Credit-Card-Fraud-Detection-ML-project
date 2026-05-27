import os
from dataclasses import dataclass, field
from src.constants.app_constants import (
    ARTIFACT_DIR,
    TIMESTAMP,
    PIPELINE_NAME,
    COLLECTION_NAME,
    DATABASE_NAME,
    PREPROCESSING_OBJECT_FILE_NAME,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    RAW_FILE_NAME,
    TRAIN_TEST_SPLIT_RATIO,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_VALID_DIR,
    DATA_VALIDATION_INVALID_DIR,
    DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    DATA_TRANSFORMATION_TRAIN_FILE_PATH,
    DATA_TRANSFORMATION_TEST_FILE_PATH,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD,
)


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    model_dir=os.path.join("final_model")
    timestamp: str=TIMESTAMP
    

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion"
    )
    feature_store_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion", "feature_store", RAW_FILE_NAME
    )
    training_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion", "ingested", TRAIN_FILE_NAME
    )
    testing_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir, "data_ingestion", "ingested", TEST_FILE_NAME
    )
    train_test_split_ratio: float = TRAIN_TEST_SPLIT_RATIO
    database_name: str = DATABASE_NAME
    collection_name: str = COLLECTION_NAME


@dataclass
class DataValidationConfig:
    schema_file_path: str
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
    )
    valid_data_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_VALID_DIR
    )
    invalid_data_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_INVALID_DIR
    )
    drift_report_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_DRIFT_REPORT_DIR
    )
    drift_report_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR_NAME,
        DATA_VALIDATION_DRIFT_REPORT_DIR,
        DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    )


@dataclass
class DataTransformationConfig:
    transformed_train_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR_NAME,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TRAIN_FILE_PATH,
    )
    transformed_test_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR_NAME,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TEST_FILE_PATH,
    )
    preprocessor_object_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR_NAME,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJECT_FILE_NAME,
    )

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_TRAINER_TRAINED_MODEL_NAME,
    )
    expected_score: float = MODEL_TRAINER_EXPECTED_SCORE
    overfitting_underfitting_threshold: float = MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD