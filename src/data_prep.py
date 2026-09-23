import os
import shutil
import glob
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

# Config
DATASET_NAME = "andrewmvd/face-mask-detection"
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
TEST_SIZE = 0.15
VAL_SIZE = 0.15

def download_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset downloaded to: {path}")
    
    # Move files to our raw directory
    os.makedirs(RAW_DIR, exist_ok=True)
    for folder in ["annotations", "images"]:
        src_folder = os.path.join(path, folder)
        dest_folder = os.path.join(RAW_DIR, folder)
        if os.path.exists(src_folder) and not os.path.exists(dest_folder):
            shutil.copytree(src_folder, dest_folder)
            print(f"Copied {folder} to {RAW_DIR}")
    print("Dataset ready in data/raw/")

def parse_and_crop():
    print("Parsing XML and cropping faces...")
    for cls in CLASSES:
        os.makedirs(os.path.join(PROCESSED_DIR, cls), exist_ok=True)
        
    xml_files = glob.glob(os.path.join(RAW_DIR, "annotations", "*.xml"))
    
    metadata = []
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find("filename").text
        image_path = os.path.join(RAW_DIR, "images", filename)
        
        if not os.path.exists(image_path):
            continue
            
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASSES:
                continue
                
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            # Crop
            crop_img = img[ymin:ymax, xmin:xmax]
            
            # Save crop
            base = os.path.splitext(filename)[0]
            crop_filename = f"{base}_{xmin}_{ymin}.jpg"
            save_path = os.path.join(PROCESSED_DIR, name, crop_filename)
            
            if crop_img.size > 0:
                cv2.imwrite(save_path, crop_img)
                
                metadata.append({
                    "filename": crop_filename,
                    "class": name,
                    "path": save_path,
                    "orig_width": img.shape[1],
                    "orig_height": img.shape[0],
                    "box_width": xmax - xmin,
                    "box_height": ymax - ymin
                })
                
    df = pd.DataFrame(metadata)
    print("\nClass Distribution after cropping:")
    print(df["class"].value_counts())
    return df

def split_data(df):
    print("\nCreating Train/Val/Test splits (70/15/15)...")
    
    # First split to get Train and Temp (30%)
    train_df, temp_df = train_test_split(df, test_size=(TEST_SIZE + VAL_SIZE), stratify=df["class"], random_state=42)
    
    # Split Temp into Val and Test (50% each of the 30% -> 15% / 15%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["class"], random_state=42)
    
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    download_dataset()
    df = parse_and_crop()
    split_data(df)
    print("\nData preparation complete!")
