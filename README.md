# Handwritten Digit Classification using Support Vector Machine (SVM)

## Overview
This project uses a Support Vector Machine (SVM) classifier from scikit-learn to recognize handwritten digits from the built-in `load_digits` dataset.

The script loads the dataset, splits it into training and testing sets, trains an SVM with a polynomial kernel, predicts the labels of the test examples, and prints the classification accuracy.

## Dataset
The project uses `sklearn.datasets.load_digits`, a built-in dataset containing 1,797 grayscale images of handwritten digits from 0 to 9. Each image is 8 × 8 pixels and is represented by 64 numeric features.

## Workflow
1. Load the digits dataset.
2. Separate the image features (`X`) and labels (`y`).
3. Split the data into training and testing subsets using an 80/20 split and `random_state=42`.
4. Create an SVM classifier using `SVC(C=1, kernel="poly")`.
5. Train the model on the training subset.
6. Predict digit labels for the test subset.
7. Calculate and print test accuracy using `accuracy_score`.

## Model details
- **Algorithm:** Support Vector Machine (SVM)
- **Library:** scikit-learn
- **Kernel:** Polynomial (`poly`)
- **C parameter:** `1`
- **Train/test split:** 80% / 20%
- **Random state:** `42`

## Requirements
- Python 3
- scikit-learn

Install the dependency:

```bash
python -m pip install -r requirements.txt
```

## Run the project

```bash
python SVM.py
```

The script prints the accuracy measured on the held-out test split. The exact value is produced when you run the script; it is not hard-coded.

## Files
- `SVM.py` — data loading, model training, prediction, and evaluation.
- `README.md` — project description and usage instructions.
- `requirements.txt` — required Python package.

## Notes
This is a compact example of using SVM for multi-class digit classification. It uses scikit-learn's built-in 8 × 8 digits dataset, not the MNIST dataset.
