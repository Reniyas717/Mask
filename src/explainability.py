import os
import cv2
import numpy as np
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries

def get_gradcam_heatmap(model, img_array, class_index, layer_name=None):
    """
    Generates a Grad-CAM heatmap for a given image and class.
    If layer_name is None, it tries to find the last convolutional layer.
    """
    if layer_name is None:
        for layer in reversed(model.layers):
            try:
                # Find the last layer that outputs a 4D tensor (e.g. Conv2D or a base_model before pooling)
                if len(layer.output.shape) == 4:
                    layer_name = layer.name
                    break
            except Exception:
                pass
                    
    if layer_name is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")
        
    # Safely get model input (handles both Sequential and Functional models in Keras 3)
    try:
        model_inputs = model.input
    except AttributeError:
        model_inputs = model.layers[0].input
        
    # Create Grad-CAM model using input and the selected convolutional output
    # We use model.layers[-1].output to avoid Sequential Keras output AttributeErrors
    grad_model = tf.keras.models.Model(
        inputs=model_inputs, 
        outputs=[model.get_layer(layer_name).output, model.layers[-1].output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap onto the original image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image {img_path}")
        
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    
    # Convert BGR to RGB for Streamlit/Matplotlib
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

def get_lime_explanation(model, img_path, class_index, img_size=(224, 224)):
    """
    Generates a LIME explanation overlay.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    
    # Needs batch dimension for predictor
    def predict_fn(images):
        return model.predict(images)
        
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_resized.astype('double'), 
        predict_fn, 
        top_labels=3, 
        hide_color=0, 
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        class_index, 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    img_boundry = mark_boundaries(temp/255.0, mask)
    return (img_boundry * 255).astype(np.uint8)
