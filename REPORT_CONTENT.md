# Face Mask Compliance Monitoring - Project Report

## 1. Problem Statement
In public spaces, manual mask-compliance monitoring is slow, inconsistent, and potentially unsafe during outbreaks. Human monitoring simply does not scale effectively for large crowds or extended periods. 

This project proposes a deep-learning computer-vision solution: a CNN-based classifier pipeline capable of running on live video. The objective is to classify each detected face into three compliance states (**Correctly Worn**, **Incorrectly Worn**, and **No Mask**) in real-time. By benchmarking multiple CNN architectures, we identify the optimal balance between accuracy, inference speed, and model size for real-world deployment.

## 2. Dataset Description
- **Source:** Kaggle "Face Mask Detection" by Andrew Mvd
- **URL:** https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **Size:** 853 source images, resulting in over 4,000 cropped faces.
- **Classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`

The dataset is highly imbalanced, with `mask_weared_incorrect` comprising a minority of the samples. Annotations were provided in PASCAL VOC XML format.

## 3. Preprocessing Theory
1. **Face Cropping:** Ground-truth bounding boxes were extracted from XML files. The regions of interest (ROI) were cropped using OpenCV to isolate the facial features, converting a detection problem into a pure classification problem.
2. **Resizing:** All cropped faces were resized to 224x224 pixels. This size was chosen to match the input requirements of ResNet50 and MobileNetV2, ensuring the network has enough spatial resolution to distinguish subtle features like the edge of a mask below the nose.
3. **Data Augmentation:** Applied exclusively to the training set to prevent overfitting and improve generalization on the minority class. 
   - *Horizontal Flip:* Included, as faces are generally symmetric.
   - *Rotation & Jitter:* Included to simulate head tilts and varying camera lighting.
   - *Vertical Flip:* Excluded, as an upside-down face is not a realistic real-world input for this system.
4. **Class Weighting:** Implemented during training using `sklearn.utils.class_weight.compute_class_weight` to penalize the model more heavily for misclassifying the minority class (`mask_weared_incorrect`).

## 4. Architectures Built
We trained and compared three models:

1. **Baseline CNN:** A custom, lightweight Convolutional Neural Network built from scratch. It consists of three Conv2D layers with ReLU activations, interleaved with MaxPooling layers to reduce spatial dimensions, followed by a dense classifier head with Dropout (0.5) for regularization. This serves as the anchor to prove the efficacy of deep learning vs. transfer learning.
2. **MobileNetV2 (Transfer Learning A):** A model optimized for mobile and edge devices. It utilizes depthwise separable convolutions to drastically reduce parameter count while maintaining accuracy. We froze the pretrained ImageNet base and fine-tuned a custom dense head.
3. **ResNet50 (Transfer Learning B):** A deep, high-capacity model using residual skip connections to mitigate the vanishing gradient problem. This model serves as the "maximum accuracy" benchmark to compare against MobileNetV2's speed.

## 5. Evaluation Metrics
Models were evaluated on a strict 15% held-out test split using the following metrics:
- **Accuracy:** Overall correctness across all classes.
- **Macro F1, Precision, Recall:** Averaged equally across all classes to prevent the majority class from masking poor performance on the minority `mask_weared_incorrect` class.
- **Inference Latency (ms) & FPS:** Measured by passing the test set through the model to simulate real-world throughput on the target hardware.
- **Model Size (MB) & Parameter Count:** Crucial for determining edge-device deployment feasibility.

## 6. Conclusion
### Key Findings
Transfer learning models significantly outperformed the Baseline CNN. MobileNetV2 emerged as the optimal choice for real-time video inference, providing a high FPS with minimal accuracy degradation compared to ResNet50.

### Challenges
The primary challenge was the severe class imbalance and the variance in lighting/resolution. Small faces captured from a distance lacked the pixel density required for the model to confidently distinguish between a bare face and a poorly worn mask.

### Future Scope
Future improvements include transitioning from a two-stage Haar Cascade + CNN pipeline to a unified single-stage object detector (e.g., YOLOv8). Furthermore, expanding the dataset using RMFD (Real-World Masked Face Dataset) would dramatically improve the model's recall on the minority class.
