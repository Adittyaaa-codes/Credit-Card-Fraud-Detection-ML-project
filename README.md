# Credit Card Fraud Detection

This project builds an end-to-end machine learning system to detect fraudulent credit card transactions. It covers everything from raw data ingestion to a live deployed web application, and the entire process is automated using GitHub Actions so that any new code pushed to the repository automatically triggers a fresh build and deployment.

---

## How the Data Gets In (ETL)

The raw credit card transaction data is first loaded using a custom ETL script. This script reads the dataset and pushes every record into a MongoDB Atlas collection. MongoDB acts as the central data store for this project, meaning all downstream pipeline steps pull from the same source of truth rather than from local CSV files. This makes the system more realistic and production ready because the data lives in a proper database, not just a flat file on disk.

---

## The Machine Learning Pipeline

The pipeline is broken into four sequential steps where each step depends on the output (called an artifact) produced by the previous one.

### Step 1: Data Ingestion

The pipeline starts by connecting to MongoDB Atlas and pulling all the transaction records from the database. It then splits the data into a training set and a test set and saves them locally as artifacts. The next step cannot run without these files being present.

### Step 2: Data Validation

Once the data is ingested, a validation check is performed on both the training and test files. It checks that the column names and data types match the expected schema defined in the project. It also runs a statistical drift check to detect whether the data distribution has shifted significantly. If validation fails, the pipeline stops early and raises an error rather than training on bad data. This step depends entirely on the artifact produced by data ingestion.

### Step 3: Data Transformation

After the data passes validation, transformation begins. A preprocessing pipeline is built using KNN Imputation to handle any missing values and standard scaling to normalize the feature values. The preprocessor is fitted on the training data and then applied to both the training and test sets. The transformed arrays and the fitted preprocessor object are saved as artifacts. The next step depends on these saved arrays.

### Step 4: Model Training

The transformed training data is used to train an XGBoost classifier. A grid search is run over the following hyperparameter combinations to find the best model:

| Hyperparameter | Values Tried |
|---|---|
| n_estimators | 50, 100 |
| max_depth | 3, 7 |
| learning_rate | 0.01, 0.1 |

The best performing model from the grid search is evaluated on both the training and test sets. Here are the results from the latest training run:

| Metric | Train | Test |
|---|---|---|
| F1 Score | 0.8994 | 0.8698 |
| Precision | 0.9705 | 0.9752 |
| Recall | 0.8380 | 0.7850 |

The high precision score (97.5% on test data) means that when the model flags a transaction as fraudulent, it is almost always correct, which is critical in a real-world fraud detection scenario.

---

## How the Models Are Saved to AWS S3

Once training finishes, both the trained model and the fitted preprocessor are automatically synced to an AWS S3 bucket. They are stored under a timestamped folder so every training run creates a new version and nothing gets overwritten. The prediction API retrieves the latest model files from this S3 bucket every time it starts up, so there is no need to manually copy any files around.

---

## The FastAPI Backend

A FastAPI application serves as the backend of the system. It exposes two main endpoints:

A GET request to `/train` triggers the full training pipeline from scratch, pulling fresh data from MongoDB, training a new model, and pushing the result to S3.

A POST request to `/predict` accepts a CSV file of transactions, loads the latest model from S3, runs the data through the preprocessor and the XGBoost model, and returns a JSON response listing which transactions are flagged as fraudulent.

---

## Deployment with Docker and AWS ECR

The entire application is containerised using Docker. The Dockerfile packages the Python environment, all dependencies, and the application code into a single image. This image is built and pushed to AWS Elastic Container Registry (ECR), which is Amazon's private Docker image repository. When the EC2 server receives a new deployment, it pulls the latest image from ECR and runs it as a container, exposing the FastAPI service on port 8080.

---

## Automated CI/CD with GitHub Actions

The whole build and deployment process is automated using a GitHub Actions workflow. Every time new code is pushed to the master branch, the following happens automatically without any manual steps:

1. The code is checked out and linted.
2. A new Docker image is built and pushed to AWS ECR.
3. The self-hosted GitHub Actions runner on the EC2 instance pulls the new image.
4. The old container is stopped and removed.
5. A fresh container is started from the new image with all the required environment variables injected as secrets.

This means that pushing code to GitHub is the only action needed to update the live production application running on AWS.
