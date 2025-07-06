# Mushroom Classification Project

## Overview
This project implements and compares three classification models — Support Vector Machine (SVM), Naïve Bayes (NB), and Deep Neural Network (DNN) — to classify mushrooms as edible or poisonous. The dataset contains both numerical and categorical features suitable for multi-class classification.

---

## Dataset

- **Source:** Kaggle Mushroom Dataset
- **Samples:** 8124
- **Features:** Mix of categorical and numerical attributes describing mushroom characteristics
- **Target:** Mushroom edibility (edible = 0, poisonous = 1)

---

## Data Preprocessing

- Handled missing values by removing or imputing.
- Encoded categorical variables using Label Encoding / One-Hot Encoding.
- Normalized numerical features.
- Split dataset into 80% training and 20% testing sets.
- Addressed class imbalance using SMOTE to augment minority class samples.

---

## Model Implementation

1. **Support Vector Machine (SVM)**
   - Kernel used: Radial Basis Function (RBF)
   - Reason: RBF handles non-linear relationships well in feature space.
   
2. **Naïve Bayes (NB)**
   - GaussianNB implemented for probabilistic classification.
   - Performs well on features that are conditionally independent.

3. **Deep Neural Network (DNN)**
   - Architecture: Input layer matching number of features, two hidden layers with ReLU activation, output layer with Softmax for multi-class.
   - Optimizer: Adam for faster convergence.
   - Trained for 20 epochs with batch size 32.
   - Training accuracy and loss curves plotted to visualize learning progress.
   - Explained backpropagation: gradients computed and used to update weights minimizing loss.

---

## Model Evaluation

- Metrics used: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- SVM showed high recall but lower precision.
- Naïve Bayes had moderate performance due to feature dependencies.
- DNN achieved best overall performance with high accuracy and balanced precision/recall.
- Confusion matrices visualized to analyze model errors.
- Training curves indicated good convergence and learning behavior for DNN.

---

## Results

- Best model: Deep Neural Network with ~98% accuracy.
- Important note: High recall for poisonous mushrooms is critical for safety.
- Visualization of training accuracy and loss provided.
- Recommendations for improving precision and reducing false positives.

---

## Usage

1. Download or clone the project files (Jupyter notebooks and dataset) to your local machine.
2. Open a terminal (Command Prompt / Bash) and navigate to the project folder:
3. Launch Jupyter Notebook: (This will open Jupyter in your default web browser.)
4. In the Jupyter interface, open the file `model_training.ipynb`.
5. Run each cell sequentially:
- Use the **Run** button or press `Shift + Enter` to execute each cell.
- This will train the models, evaluate them, and display results inline.
6. View the printed evaluation metrics, confusion matrices, and training accuracy/loss plots within the notebook.
---

## Dependencies

- Python 3
- numpy
- pandas
- scikit-learn
- imblearn
- tensorflow / keras
- matplotlib
- seaborn

---

## Code Repository

https://github.com/uchithvihanga/mashroom_classification_project.git

---

## Conclusion

This project demonstrates effective classification of mushrooms using classical ML and deep learning models. DNN performed best, but further work can explore hyperparameter tuning and additional feature engineering for improved safety and precision.

*Prepared by: Uchith Vihanga*  
*Date: 2025-07-06*  
