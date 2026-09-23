import tensorflow as tf
from tensorflow.keras import layers, models, applications

IMG_SIZE = (224, 224)
NUM_CLASSES = 3

def build_baseline_cnn(input_shape=(224, 224, 3)):
    """
    A simple custom CNN trained from scratch.
    Serves as a baseline to demonstrate the value of transfer learning.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Normalization (scaling to [0,1]) since it's from scratch
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax', name='classifier_head')
    ], name="baseline_cnn")
    
    return model

def build_mobilenetv2(input_shape=(224, 224, 3)):
    """
    Transfer learning model A.
    Chosen for real-time inference speed (edge computing friendly).
    Frozen base, trained head.
    """
    # MobileNetV2 expects inputs in [-1, 1], so we use its preprocess_input
    inputs = layers.Input(shape=input_shape)
    x = applications.mobilenet_v2.preprocess_input(inputs)
    
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='classifier_head')(x)
    
    model = models.Model(inputs, outputs, name="mobilenet_v2")
    return model

def build_resnet50(input_shape=(224, 224, 3)):
    """
    Transfer learning model B.
    Higher capacity model for benchmarking against the lightweight MobileNet.
    Base is frozen initially, though could be unfreezed for fine-tuning.
    """
    # ResNet50 expects inputs in BGR [-123.68, ..., ...] style scaling
    inputs = layers.Input(shape=input_shape)
    x = applications.resnet50.preprocess_input(inputs)
    
    base_model = applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='classifier_head')(x)
    
    model = models.Model(inputs, outputs, name="resnet50")
    return model

def get_model(model_name):
    if model_name == "baseline_cnn":
        return build_baseline_cnn()
    elif model_name == "mobilenetv2":
        return build_mobilenetv2()
    elif model_name == "resnet50":
        return build_resnet50()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
