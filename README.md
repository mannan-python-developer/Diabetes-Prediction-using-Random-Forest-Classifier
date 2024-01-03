# Diabetes Prediction using Random Forest Classifier

This Jupyter Notebook contains a Python script for predicting diabetes using a Random Forest Classifier. The dataset used for this analysis includes features such as pregnancies, glucose levels, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age.

## Dataset Overview

The dataset, sourced from a CSV file, is loaded into a Pandas DataFrame. It includes 768 instances with features related to health parameters and an 'Outcome' column indicating the presence (1) or absence (0) of diabetes.

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'G:\Machine Learning\Datasets\diabetes.csv')
```
## Data Splitting

The dataset undergoes a split into training and testing sets using the `train_test_split` function from Scikit-Learn. Approximately 80% of the data is allocated for training the Random Forest Classifier, while the remaining 20% is reserved for testing.

```python
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']], df["Outcome"], test_size=0.2)
```

## Random Forest Classifier

The Random Forest Classifier, a robust ensemble learning method, is implemented and trained using Scikit-Learn's  `RandomForestClassifier` class.

```python
from sklearn.ensemble import RandomForestClassifier

# Instantiate and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

## Model Prediction

The trained model is employed to make predictions on the test set.

```python
predictions = model.predict(X_test)
print(predictions)
```

## Model Evaluation

Model accuracy is assessed using the `score` method, which compares the predicted outcomes with the actual outcomes in the test set.

```python
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
```

The model achieves a testing accuracy of approximately 77.92%.


This notebook serves as an illustrative example of implementing a Random Forest Classifier for diabetes prediction. It highlights the workflow from data splitting to model evaluation, providing a foundation that users can adapt for similar classification tasks.
