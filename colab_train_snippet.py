"""
COPY AND PASTE THE ENTIRE CONTENT OF THIS FILE INTO A SINGLE GOOGLE COLAB CELL AND RUN IT.
It will:
1. Install requirements.
2. Download the dataset from Kaggle.
3. Process and crop the faces.
4. Train all 3 models (Baseline, MobileNetV2, ResNet50).
5. Zip the trained models and training metrics into a file you can download.
"""

import os
print("Installing dependencies...")
os.system("pip install kagglehub opencv-python scikit-learn pandas matplotlib -q")

import shutil
import glob
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

# --- 1. CONFIG ---
DATASET_NAME = "andrewmvd/face-mask-detection"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs/figures"
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 2. DATA PREP ---
print("\n--- 2. DOWNLOADING DATA ---")
path = kagglehub.dataset_download(DATASET_NAME)
for folder in ["annotations", "images"]:
    src_folder = os.path.join(path, folder)
    dest_folder = os.path.join(RAW_DIR, folder)
    if os.path.exists(src_folder) and not os.path.exists(dest_folder):
        shutil.copytree(src_folder, dest_folder)

print("\n--- 3. PARSING & CROPPING ---")
for cls in CLASSES:
    os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)
xml_files = glob.glob(os.path.join(RAW_DIR, "annotations", "*.xml"))
metadata = []
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    image_path = os.path.join(RAW_DIR, "images", filename)
    if not os.path.exists(image_path): continue
    img = cv2.imread(image_path)
    if img is None: continue
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES: continue
        bndbox = obj.find("bndbox")
        xmin, ymin = int(bndbox.find("xmin").text), int(bndbox.find("ymin").text)
        xmax, ymax = int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)
        crop_img = img[ymin:ymax, xmin:xmax]
        if crop_img.size > 0:
            crop_filename = f"{os.path.splitext(filename)[0]}_{xmin}_{ymin}.jpg"
            save_path = os.path.join(PROCESSED_DIR, name, crop_filename)
            cv2.imwrite(save_path, crop_img)
            metadata.append({"filename": crop_filename, "class": name, "path": save_path})

df = pd.DataFrame(metadata)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["class"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["class"], random_state=42)
train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

# --- 4. MODELS ---
def get_model(name):
    if name == "baseline_cnn":
        return models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Rescaling(1./255),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'), layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'), layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'), layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'), layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
    elif name == "mobilenetv2":
        inputs = layers.Input(shape=(224, 224, 3))
        x = applications.mobilenet_v2.preprocess_input(inputs)
        base = applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        return models.Model(inputs, layers.Dense(3, activation='softmax')(x))
    elif name == "resnet50":
        inputs = layers.Input(shape=(224, 224, 3))
        x = applications.resnet50.preprocess_input(inputs)
        base = applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        return models.Model(inputs, layers.Dense(3, activation='softmax')(x))

# --- 5. TRAINING ---
classes = sorted(train_df["class"].unique())
class_indices = {c: i for i, c in enumerate(classes)}
with open(os.path.join(MODELS_DIR, "class_index.json"), "w") as f: json.dump(class_indices, f)

train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, brightness_range=[0.8, 1.2], zoom_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_dataframe(train_df, x_col="path", y_col="class", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", classes=classes)
val_gen = val_datagen.flow_from_dataframe(val_df, x_col="path", y_col="class", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", classes=classes, shuffle=False)

weights_arr = compute_class_weight('balanced', classes=np.array(classes), y=train_df["class"].values)
class_weights = dict(enumerate(weights_arr))

for m_name in ["baseline_cnn", "mobilenetv2", "resnet50"]:
    print(f"\n--- TRAINING {m_name.upper()} ---")
    model = get_model(m_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    cb = [ModelCheckpoint(f"models/{m_name}.h5", save_best_only=True), EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, class_weight=class_weights, callbacks=cb)
    
    plt.figure()
    plt.plot(hist.history['accuracy'], label='Train')
    plt.plot(hist.history['val_accuracy'], label='Val')
    plt.legend()
    plt.savefig(f"outputs/figures/{m_name}_acc.png")

print("\n--- ZIPPING OUTPUTS ---")
os.system("zip -r models_and_outputs.zip models/ outputs/ data/processed/*.csv")
print("\nDONE! Download models_and_outputs.zip from the Colab file explorer.")
