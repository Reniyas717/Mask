import os
import json
import argparse
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from models import get_model, IMG_SIZE, NUM_CLASSES

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUTPUTS_DIR = os.path.join("outputs", "figures")
BATCH_SIZE = 32
EPOCHS = 20

def plot_history(history, model_name):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{model_name}_accuracy.png"))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{model_name}_loss.png"))
    plt.close()

def main(model_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data manifests
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    
    # Save class mapping
    classes = sorted(train_df["class"].unique())
    class_indices = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(MODELS_DIR, "class_index.json"), "w") as f:
        json.dump(class_indices, f)
        
    # Data Augmentation (ONLY on Train)
    # Justification: Masks are not vertically symmetric, so no vertical_flip. 
    # Lighting varies (brightness_range), faces rotate slightly (rotation_range).
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator() # No augmentation for validation
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="path",
        y_col="class",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=classes,
        shuffle=False
    )
    
    # Compute class weights for imbalance
    import numpy as np
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=train_df["class"].values
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Class weights computed: {class_weights}")
    
    # Get model
    model = get_model(model_name)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    print(f"Training {model_name}...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Plot & save history
    plot_history(history, model_name)
    print(f"Training complete. Best model saved to {model_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["baseline_cnn", "mobilenetv2", "resnet50"])
    args = parser.parse_args()
    main(args.model)
