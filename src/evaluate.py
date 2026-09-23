import os
import time
import json
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

from models import IMG_SIZE, get_model

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUTPUTS_FIG = os.path.join("outputs", "figures")
OUTPUTS_METRICS = os.path.join("outputs", "metrics")
BATCH_SIZE = 32

def count_params(model):
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return trainable_count + non_trainable_count

def evaluate_model(model_name, test_df, class_indices):
    print(f"\nEvaluating {model_name}...")
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found. Skipping.")
        return None
        
    try:
        model = get_model(model_name)
        model.load_weights(model_path)
    except Exception as e:
        print(f"Failed to load weights for {model_name}: {e}")
        return None
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    param_count = count_params(model)
    
    classes = list(class_indices.keys())
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="path",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        shuffle=False
    )
    
    # Latency / FPS calculation (single batch warmup, then average over remaining)
    # Get one batch for warmup
    sample_batch = next(iter(test_generator))[0]
    _ = model.predict(sample_batch) # warmup
    
    start_time = time.time()
    predictions = model.predict(test_generator)
    end_time = time.time()
    
    total_images = len(test_df)
    latency_ms = ((end_time - start_time) / total_images) * 1000
    fps = 1000 / latency_ms
    
    y_true = test_generator.classes
    y_pred = predictions.argmax(axis=1)
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_FIG, f"{model_name}_cm.png"))
    plt.close()
    
    # ROC / AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUTS_FIG, f"{model_name}_roc.png"))
    plt.close()
    
    results = {
        "Model": model_name,
        "Accuracy": report['accuracy'],
        "Macro F1": report['macro avg']['f1-score'],
        "Macro Precision": report['macro avg']['precision'],
        "Macro Recall": report['macro avg']['recall'],
        "Size (MB)": round(model_size_mb, 2),
        "Parameters": param_count,
        "Latency (ms)": round(latency_ms, 2),
        "FPS": round(fps, 1)
    }
    
    # Add per-class recall (important for minority class)
    for cls in classes:
        results[f"Recall ({cls})"] = report[cls]['recall']
        
    return results

def main():
    os.makedirs(OUTPUTS_FIG, exist_ok=True)
    os.makedirs(OUTPUTS_METRICS, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    with open(os.path.join(MODELS_DIR, "class_index.json"), "r") as f:
        class_indices = json.load(f)
        
    models_to_eval = ["baseline_cnn", "mobilenetv2", "resnet50"]
    all_results = []
    
    for m in models_to_eval:
        res = evaluate_model(m, test_df, class_indices)
        if res:
            all_results.append(res)
            
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUTS_METRICS, "model_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        print("\nEvaluation Complete! Summary Table:")
        print(comparison_df.to_string(index=False))
        
if __name__ == "__main__":
    main()
