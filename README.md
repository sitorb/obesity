# Obesity Prediction using Machine Learning

## Overview

This project aims to predict obesity levels using machine learning algorithms. It involves data preprocessing, model training, evaluation, and visualization of results. The project utilizes various classification algorithms and compares their performance to identify the most suitable model for obesity prediction.

## Code Structure

The project code is structured into the following sections:

1. **Imports:** Importing necessary libraries for data manipulation, model training, and evaluation.
2. **Data Loading:** Loading the obesity dataset from a CSV file using Pandas.
3. **Data Preprocessing:** Handling missing values and encoding categorical variables using Label Encoding.
4. **Data Splitting:** Splitting the data into training and testing sets using `train_test_split`.
5. **Feature Scaling:** Standardizing features using `StandardScaler`.
6. **Model Training and Evaluation:** Training and evaluating various classification models, including:
    - Gaussian Naive Bayes
    - Softmax Regression
    - Random Forest
    - Support Vector Machine
7. **Model Comparison:** Comparing the performance of trained models based on accuracy.
8. **Visualizations:** Creating visualizations to gain insights into the data and model results, including:
    - Feature importance plot (for Random Forest)
    - Confusion matrices for each model
    - Distribution of obesity levels
    - Correlation matrix

## Logic

The project follows a typical machine learning workflow:

1. **Data Preparation:** Clean and prepare the data for model training.
2. **Model Selection:** Choose appropriate machine learning algorithms for the prediction task.
3. **Model Training:** Train the selected models using the training data.
4. **Model Evaluation:** Evaluate the performance of the trained models using the testing data.
5. **Model Comparison:** Compare the performance of different models and select the best one.
6. **Visualization:** Visualize the results to gain insights and communicate findings.

![image](https://github.com/user-attachments/assets/474a4037-8102-4991-a63c-ce9453014ecf)

![image](https://github.com/user-attachments/assets/1e209d75-e2bd-4fd6-aed3-7075908edc59)


## Technology

The project utilizes the following technologies:

- **Python:** The primary programming language used for data analysis and machine learning.
- **Pandas:** A library for data manipulation and analysis.
- **NumPy:** A library for numerical computing.
- **Scikit-learn:** A library for machine learning algorithms and tools.
- **Seaborn and Matplotlib:** Libraries for data visualization.

## Algorithms

The project employs the following machine learning algorithms:

- **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem.
- **Softmax Regression:** A generalization of logistic regression for multi-class classification.
- **Random Forest:** An ensemble learning method that combines multiple decision trees.
- **Support Vector Machine (SVM):** A powerful algorithm for classification and regression tasks.

## Conclusion

This project demonstrates the application of machine learning techniques for obesity prediction. The results can be used to gain insights into the factors contributing to obesity and develop strategies for prevention and intervention.
