# Detection of Pneumonia Using Chest X-Ray Images

## Project Description

This repository contains three notebooks implementing deep learning models to classify chest X-ray images as either normal or indicative of pneumonia. The project compares three distinct approaches:

1. **Base Model**: A custom convolutional neural network (CNN) built from scratch.
2. **Transfer Learning Model**: A pre-trained ResNet-50 model fine-tuned for the task.
3. **Teacher-Student Model**: A knowledge distillation approach where a smaller student network learns from a larger teacher network.

The goal is to evaluate these models for accuracy, efficiency, and suitability for medical diagnostics, using the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle.

## Files in the Repository

1. **Base_model.ipynb**: Implements a custom CNN using TensorFlow and Keras. It includes data preprocessing, model training, and evaluation.
2. **Transfer_model.ipynb**: Uses transfer learning with ResNet-50 in PyTorch, fine-tuned for binary classification.
3. **student teacher model.ipynb**: Implements a teacher-student model with knowledge distillation in PyTorch, aiming for a lightweight yet effective solution.

## Comparison of the Three Models

| Model                  | Framework         | Description                                      | Purpose                              |
|-----------------------|-------------------|--------------------------------------------------|--------------------------------------|
| Base Model            | TensorFlow/Keras  | 11-layer custom CNN with conv, pooling, dropout  | Baseline model built from scratch    |
| Transfer Model        | PyTorch           | Pre-trained ResNet-50, fine-tuned for task       | Leverages transfer learning          |
| Teacher-Student Model | PyTorch           | Small student CNN learns from a larger teacher   | Balances accuracy and efficiency     |

### Methodology

#### Dataset
- **Source**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Structure**: Folders: `train`, `val`, `test`; Subfolders: `NORMAL`, `PNEUMONIA`.
- **Preprocessing**:
  - **Base Model**: Resized to 224x224, normalized, augmented (e.g., flips, rotations).
  - **Transfer Model**: Resized, normalized to ImageNet standards, augmented using `torchvision.transforms`.
  - **Teacher-Student Model**: Similar to Transfer Model for consistency.

#### Model Architectures
1. **Base Model**:
   - 11 layers: convolutional (e.g., 32, 64 filters), max-pooling, dropout (e.g., 0.25), dense layer with sigmoid.
   - Trained from scratch.
2. **Transfer Model**:
   - ResNet-50 (pre-trained on ImageNet), final layer replaced for binary output.
   - Fine-tuned with some layers unfrozen.
3. **Teacher-Student Model**:
   - **Teacher**: Likely ResNet-50 (fine-tuned).
   - **Student**: Smaller custom CNN (fewer layers/filters).
   - Uses knowledge distillation with a combined loss (soft labels from teacher + ground truth).

#### Training
- **Base Model**: Binary cross-entropy loss, Adam optimizer, metrics: accuracy, recall.
- **Transfer Model**: Cross-entropy loss, SGD or Adam optimizer, fine-tuned with a lower learning rate.
- **Teacher-Student Model**: Distillation loss (temperature-controlled soft labels) + cross-entropy, trained to mimic teacher.

### Results
*(Note: Replace placeholders with your actual results from the notebooks.)*
- **Base Model**: Achieved 91.19% test accuracy, with high AUC-ROC for pneumonia detection.
- **Transfer Model**: Achieved 87.50% test accuracy, benefiting from pre-trained features.
- **Teacher-Student Model**: Achieved 81.09% test accuracy, with reduced complexity (fewer parameters) and focus on recall for medical use.

The Base Model provides a strong baseline, the Transfer Model leverages pre-trained knowledge for robust performance, and the Teacher-Student Model offers a lightweight alternative suitable for resource-constrained environments.

## Dependencies

- **Base_model.ipynb**: TensorFlow, Keras, NumPy, pandas, OpenCV, scikit-learn, matplotlib, seaborn.
- **Transfer_model.ipynb** & **student teacher model.ipynb**: PyTorch, torchvision, NumPy, matplotlib, scikit-learn.

