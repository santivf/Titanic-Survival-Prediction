# Titanic Survival Prediction

This project predicts passenger survival on the Titanic using two classification models:
- **Logistic Regression**
- **Decision Tree** (with entropy criterion and max_depth=4 for interpretability)

The models are trained and evaluated on a preprocessed version of the classic Titanic dataset, comparing accuracy and confusion matrices. A visualized decision tree is also generated.

## Dataset
- File: `titanic_processed.csv` (included in the repository)
- Source: Preprocessed version of the public Titanic dataset (commonly used for classification tasks)
- Features: Passenger attributes (e.g., Age, Fare, Pclass, Sex, Embarked, etc. – already encoded)
- Target: `Survived` (binary: 0 = No, 1 = Yes)

## What the Script Does
1. Loads the preprocessed dataset
2. Splits data into train/test sets (80/20)
3. Trains a Logistic Regression model
4. Trains a Decision Tree Classifier (entropy, max_depth=4)
5. Evaluates both models with:
   - Accuracy score
   - Confusion matrix (displayed as plot)
6. Visualizes the full Decision Tree structure

## Results (example output)
Precision Logistic: 0.XX

Precision Tree: 0.XX

Confusion matrices and the decision tree plot are displayed for interpretability and comparison.

## Tools
- Python
- Pandas – data loading and manipulation
- Scikit-learn – models (Logistic Regression, Decision Tree), metrics, and tree visualization
- Matplotlib – plotting confusion matrices and decision tree

## How to Run
1. Clone the repository:
git clone https://github.com/santivf/titanic-survival-prediction.git

2. Install dependencies:
pip install -r requirements.txt

3. Run the script:
python titanic_survival_prediction.py
