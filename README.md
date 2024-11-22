# Diabetes Prediction Using Support Vector Machine (SVM)

This project aims to predict whether a person is diabetic based on various health-related features using the **PIMA Diabetes Dataset** and a **Support Vector Machine (SVM)** classifier.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Installation Guide](#installation-guide)
5. [Usage](#usage)
6. [Model Evaluation](#model-evaluation)
7. [License](#license)

## Project Overview

The project uses the PIMA Diabetes dataset to train a machine learning model to predict diabetes. The dataset includes several features such as age, BMI, glucose levels, and other health metrics. The model utilizes a **Support Vector Machine** classifier with a **linear kernel** for prediction.

### Objectives:
- To load and preprocess the dataset.
- To train a model using **SVM** for binary classification (Diabetic or Non-diabetic).
- To evaluate model accuracy on both training and testing data.
- To create a simple predictive system for new input data.

## Technologies Used

- **Python**: Programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical computations.
- **Scikit-learn**: For machine learning, including the SVM classifier and data preprocessing (StandardScaler).
- **Jupyter Notebook**: For creating an interactive environment for code and results.

## Dataset

The **PIMA Diabetes Dataset** contains 768 rows and 9 columns, with the following attributes:
- `Pregnancies`: Number of pregnancies the person has had.
- `Glucose`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
- `BloodPressure`: Diastolic blood pressure (mm Hg).
- `SkinThickness`: Skinfold thickness (mm).
- `Insulin`: 2-hour serum insulin (mu U/ml).
- `BMI`: Body Mass Index.
- `DiabetesPedigreeFunction`: A function that represents the genetic likelihood of diabetes.
- `Age`: Age of the person.
- `Outcome`: Target variable (0 = Non-diabetic, 1 = Diabetic).


## Usage

1. **Loading the dataset**:
    The dataset is loaded from a CSV file:
    ```python
    diabetes_dataset = pd.read_csv('diabetes.csv')
    ```

2. **Preprocessing**:
    Standardization of data is done using `StandardScaler` to ensure features have a mean of 0 and a standard deviation of 1.
    ```python
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    ```

3. **Model Training**:
    A Support Vector Machine classifier is trained using the preprocessed data:
    ```python
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    ```

4. **Making Predictions**:
    New input data can be predicted with:
    ```python
    input_data = (10, 168, 74, 0, 0, 38, 0.537, 34)
    input_data = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data)
    prediction = classifier.predict(std_data)
    ```

    The output will indicate whether the person is diabetic or not:
    ```python
    if prediction[0] == 0:
        print("The person is not diabetic")
    else:
        print("The person is diabetic")
    ```

## Model Evaluation

The accuracy of the model is evaluated on both the training and testing datasets:

- **Training Accuracy**: ~78.66%
- **Testing Accuracy**: ~77.27%

These values reflect how well the model generalizes to unseen data.

## Author
Uhashini(https://www.linkedin.com/in/uhashini-n-3b144a291/)
