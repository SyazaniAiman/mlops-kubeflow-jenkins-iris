# Iris Classification ML Pipeline with Kubeflow  
# test
## Overview  
This repository contains a **Kubeflow Pipelines** implementation for an **Iris flower classification** task. It automates data acquisition, preprocessing, model training, and evaluation using **Kubeflow on Kubernetes**. The pipeline is modular, allowing easy customization for similar ML workflows.  

## Repository Structure  
```plaintext
kubeflow-ml-pipeline/
│── components/
│   ├── data_loader.py         # Fetches the Iris dataset
│   ├── feature_engineering.py # Prepares features and splits dataset
│   ├── model_training.py      # Trains a classification model
│   ├── evaluation.py          # Evaluates model performance
│── pipeline.py                # Defines the Kubeflow pipeline
│── iris_pipeline.yaml         # Compiled pipeline YAML for execution
│── requirements.txt           # Python dependencies
│── README.md                  # Documentation  
```

## Steps to Use This Repository

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/kubeflow-ml-pipeline.git
cd kubeflow-ml-pipeline
```

### 2. Set Up Kubeflow
Ensure Kubeflow is installed on a Kubernetes cluster. If using Google Kubernetes Engine (GKE), follow Kubeflow’s official setup guide.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Compile the Pipeline
```bash
python pipeline.py
```
This generates `iris_pipeline.yaml`, which defines the pipeline structure.

### 5. Upload to Kubeflow Pipelines UI
    1.	Open the Kubeflow UI
	2.	Navigate to Pipelines > Upload Pipeline
	3.	Select iris_pipeline.yaml and upload

### 6. Run the Pipeline
	1.	Create an Experiment and launch the pipeline
	2.	Monitor execution via the Kubeflow UI

For more details, refer to the Kubeflow documentation.
