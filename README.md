# Iris Classification ML Pipeline with Kubeflow (Assignment 2)
# Muhammad Syazaniaiman Bin Mohd Shahimi
## Overview
This repository contains a **Kubeflow Pipelines (KFP)** implementation for an **Iris flower classification** task. It automates the ML workflow, including data loading, feature processing, model training, and evaluation. The pipeline is modular, so the same structure can be adapted to other ML use cases.

This project is used for **Assignment 2** in the course **Artificial Intelligence in Software Engineering**.

## Repository Structure
```plaintext
mlops-kubeflow-jenkins-iris/
│── components/
│   ├── data_loader.py               # Loads the Iris dataset
│   ├── feature_engineering.py       # Prepares features and splits dataset
│   ├── model_training.py            # Trains a classification model
│   ├── evaluation.py                # Evaluates model performance
│── pipeline.py                      # Defines and compiles the Kubeflow pipeline
│── iris_pipeline.yaml               # Compiled pipeline YAML (ready to upload/run)
│── iris_classifier_pipeline.yaml    # Alternative compiled pipeline YAML
│── requirements.txt                 # Python dependencies
│── README.md                        # Documentation

For more details, refer to the Kubeflow documentation.
