# COSE471_data-science


**by CNN**

**1. Download the dataset (AAF)**
- [AAF Dataset](https://github.com/JingchunCheng/All-Age-Faces-Dataset.git)

**2. Install Requirements**
```bash
pip install torch torchvision shap scikit-learn tqdm pillow matplotlib numpy pandas opencv-python
```

**3. Train the Model**
```bash
python train.py
```
- Trains a CNN (ResNet18) model on the AAF dataset
- Uses weighted sampling to handle data imbalance
- Saves the best model as best_model.pth
- Logs metrics per epoch to training_metrics.csv

**4. Run Analysis (SHAP + Grad-CAM)**
```bash
python analysis.py
```
- Loads the trained model (model2.pth)
- Predicts age from an input image
- Visualizes Grad-CAM heatmap to show attention areas
- Computes SHAP values to explain feature contribution (e.g., eyes, forehead, mouth)
- Compares predicted age with true age and interprets appearance (young-looking / old-looking)

**5. Project Structure**
```bash
.
├── aaf_age_labels.csv           # CSV with image paths and age labels
├── model2.pth                   # Trained model weights
├── training_metrics(model2).csv # Training log
├── train.py                     # Training script
├── analysis.py                  # Grad-CAM + SHAP analysis script
├── make_csv.ipynb               # CSV preprocessing
├── practice.ipynb               # Development notebook
├── README.md
```
