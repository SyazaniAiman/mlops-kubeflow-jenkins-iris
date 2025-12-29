from kfp import dsl, compiler
from components.data_loader import fetch_data
from components.feature_engineering import process_features
from components.model_training import train_model
from components.evaluation import evaluate_model
# test
@dsl.pipeline(name="iris-classifier-pipeline")
def iris_pipeline():
    """Defines the complete pipeline for classification."""
    # Load dataset
    data_task = fetch_data()
    
    # Process features
    feature_task = process_features(raw_data=data_task.outputs["data_output"])
    
    # Train the model
    model_task = train_model(
        train_features=feature_task.outputs["train_features"],
        train_labels=feature_task.outputs["train_labels"]
    )
    
    # Evaluate performance
    eval_task = evaluate_model(
        test_features=feature_task.outputs["test_features"],
        test_labels=feature_task.outputs["test_labels"],
        model_file=model_task.outputs["model_output"]
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="iris_classifier_pipeline.yaml"
    )
