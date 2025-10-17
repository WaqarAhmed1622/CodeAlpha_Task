This repository contains two machine learning projects: a car price predictor and an Iris flower species classifier. Each project includes a Jupyter Notebook detailing the data analysis and model training process, the dataset used, a saved machine learning model, and a simple GUI application for making predictions.

Projects
1. Car Price Prediction
This project aims to predict the selling price of used cars based on various features.

Features:

Present_Price: The current showroom price of the car.
Driven_kms: The number of kilometers the car has been driven.
Fuel_Type: The type of fuel the car uses (Petrol, Diesel, CNG).
Selling_type: Whether the car is sold by a Dealer or an Individual.
Transmission: The transmission type (Manual, Automatic).
Owner: The number of previous owners.
Age_of_Car: The age of the car in years.
Methodology:

Data Preprocessing: The dataset (car data.csv) is loaded, and a new feature, Age_of_Car, is engineered from the Year column. Categorical features like Fuel_Type, Selling_type, and Transmission are encoded into numerical values. Outliers in the target variable (Selling_Price) are handled.
Model Training: Four different regression models were trained and evaluated:
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
Evaluation: The models were compared based on their R² score. The XGBoost Regressor achieved the highest accuracy with an R² score of approximately 87.4%.
Model Saving: The trained XGBoost model is saved as car_price_predictor using joblib.
GUI Application: A GUI built with Tkinter allows users to enter car attributes and get a predicted selling price.
Files:

Car price prediction/Car_price_prediction.ipynb: Jupyter Notebook with the complete analysis and model development.
Car price prediction/car data.csv: The dataset used for training.
Car price prediction/car_price_predictor: The saved, trained XGBoost model.
2. Iris Flower Classification
This is a classic machine learning project to classify Iris flowers into one of three species (Iris-setosa, Iris-versicolor, Iris-virginica) based on their sepal and petal dimensions.

Features:

SepalLengthCm: Length of the sepal in centimeters.
SepalWidthCm: Width of the sepal in centimeters.
PetalLengthCm: Length of the petal in centimeters.
PetalWidthCm: Width of the petal in centimeters.
Methodology:

Exploratory Data Analysis (EDA): The Iris.csv dataset is analyzed using visualizations like histograms, scatter plots, and a correlation heatmap to understand the relationships between features.
Data Preprocessing: The categorical Species column is label-encoded into numerical values (0, 1, 2).
Model Training: Two classification models were trained:
Logistic Regression
K-Neighbors Classifier (KNN)
Evaluation: Both models achieved high accuracy, with a score of approximately 95.6% on the test set.
Model Saving: The trained K-Neighbors Classifier model is saved as Iris_Flower_Classification using joblib.
GUI Application: A Tkinter-based GUI is provided to classify an Iris flower by inputting its sepal and petal measurements.
Files:

Iris Project/Iris Flower Classification.ipynb: The Jupyter Notebook containing the full workflow.
Iris Project/Iris.csv: The Iris dataset.
Iris Project/Iris_Flower_Classification: The saved, trained K-Neighbors Classifier model.
Getting Started
Prerequisites
You will need Python and the following libraries installed. You can install them using pip:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
How to Run the Applications
Clone the repository:

git clone https://github.com/WaqarAhmed1622/CodeAlpha_Task.git
cd CodeAlpha_Task
Navigate to a project directory:

For the car price predictor: cd "Car price prediction"
For the Iris classifier: cd "Iris Project"
Run the GUI: The GUI application code is included at the end of each Jupyter Notebook (Car_price_prediction.ipynb and Iris Flower Classification.ipynb). To launch an application, open the corresponding notebook, run all the cells, and the final cell will start the Tkinter GUI.

License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
