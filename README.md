# Churn Predictor: An ANN-Based Customer Retention Model

Welcome to **Churn Predictor**, a deep learning project leveraging Artificial Neural Networks (ANN) to predict customer churn based on banking data. This project demonstrates how deep learning can be applied to real-world classification problems using structured data.

---

## Problem Statement

Customer retention is critical in the banking sector. Predicting whether a customer will leave (churn) allows banks to take proactive measures. Using the **Churn_Modelling.csv** dataset, this project builds an ANN to classify customers as likely to churn or stay.

---

## Goal
Develop an accurate ANN model that predicts customer churn based on features like credit score, geography, age, balance, and more.

---

## Dataset Overview
The dataset contains customer information, including:

- **CreditScore**
- **Geography**
- **Gender**
- **Age**
- **Balance**
- **Number of Products**
- **IsActiveMember**
- **EstimatedSalary**
- **Exited** (Target Variable)

---

## Technologies & Libraries Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning Tools**: Scikit-learn
- **Callbacks**: EarlyStopping, TensorBoard

---

## Project Workflow

### 1️⃣ Data Preprocessing
- Load dataset
- Handle categorical variables using **One-Hot Encoding** and **Label Encoding**
- Feature Scaling with **StandardScaler**

### 2️⃣ Model Building
- Constructed a deep learning model using **Keras Sequential API**
- Layers include Dense layers, activation functions like ReLU, LeakyReLU, Dropout for regularization

### 3️⃣ Model Training
- Implemented **EarlyStopping** to avoid overfitting
- Tracked performance using **TensorBoard**

### 4️⃣ Evaluation
- Evaluated using metrics such as **Accuracy** and **ROC AUC Score**
- Visualized training history and confusion matrix

---

## Results
The ANN model delivered strong performance on the test dataset:

- **Accuracy Score**: **85.95%**
- Evaluated using additional metrics like ROC AUC for better classification insights.
- Proper regularization and early stopping helped prevent overfitting.

Visual performance evaluation includes:

- **Training vs Validation Accuracy**
- **Loss Curves**
- **Confusion Matrix**
- **ROC Curve**

---

## How to Run
1. Clone this repository
2. Ensure you have Python 3.x and install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Untitled0_Styled.ipynb
   ```

---

## Contributing
Feel free to fork the repo and submit pull requests for improvements!

