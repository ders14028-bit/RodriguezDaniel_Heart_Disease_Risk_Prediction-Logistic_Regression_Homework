# Heart Disease Risk Prediction: Logistic Regression Homework

## Daniel Esteban Rodriguez Suarez

## Introductory Context

Heart disease is the world’s leading cause of death, responsible for approximately **18 million deaths annually**, according to the **World Health Organization (WHO)**. Early identification of individuals at risk is critical for improving patient outcomes and optimizing healthcare resources.

In this project, logistic regression is used to predict the **risk of heart disease** based on clinical features such as age, cholesterol levels, blood pressure, and heart rate. The implementation is carried out **from scratch** using core numerical libraries, emphasizing theoretical understanding and practical application.

The dataset used is the **Heart Disease Dataset** from the UCI Machine Learning Repository, consisting of **303 patient records**, **14 clinical features**, and a **binary target variable** (1 = disease presence, 0 = absence).  
The project covers model training, visualization, regularization, and a simulated deployment pipeline using **Amazon SageMaker**.

---

## Homework Overview

This homework is implemented in a **Jupyter Notebook**, following concepts introduced in class such as:
- Sigmoid function
- Binary cross-entropy loss
- Gradient Descent optimization

Only **NumPy, Pandas, and Matplotlib** are used for core model implementation (no scikit-learn for training).

The project emphasizes:
- Exploratory data analysis
- Model interpretability
- Hyperparameter tuning
- Clear documentation and analysis

---

## Dataset Description

- **Source:** Kaggle – Heart Disease Dataset  
  https://www.kaggle.com/datasets/neurocipher/heartdisease
- **Records:** 303 patients  
- **Features:** 14 clinical attributes  
- **Target:** Heart disease presence (binary)
- **Example feature ranges:**
  - Age: 29–77 years
  - Cholesterol: 112–564 mg/dL
  - Resting Blood Pressure, Max Heart Rate, ST Depression, Number of Major Vessels
- **Class distribution:** ~55% disease presence

The dataset was downloaded manually from Kaggle and loaded as a CSV file into Pandas.

---

## Project Structure

├── heart_disease_lr_analysis.ipynb
├── README.md
├── heart.csv


---

##  Step-by-Step Implementation

### **Step 1: Load and Prepare the Dataset**
- Load dataset into Pandas
- Binarize target variable (1 = disease, 0 = no disease)
- Exploratory Data Analysis (EDA):
  - Summary statistics
  - Missing values and outlier inspection
  - Class distribution plot
- Preprocessing:
  - Stratified 70/30 train-test split
  - Feature normalization
  - Selection of at least 6 features (Age, Cholesterol, BP, Max HR, ST Depression, Vessels)
- Markdown summary of insights and preprocessing decisions

---

### **Step 2: Implement Logistic Regression from Scratch**
- Implement:
  - Sigmoid function
  - Binary cross-entropy loss
  - Gradient Descent optimization
- Train model using:
  - Learning rate ≈ 0.01
  - ≥1000 iterations
- Plot cost vs. iterations
- Predict using threshold = 0.5
- Evaluate:
  - Accuracy
  - Precision
  - Recall
  - F1-score (train and test sets)
- Report results in tables and discuss convergence and coefficient interpretations

---

### **Step 3: Visualize Decision Boundaries**
- Select **at least 3 feature pairs**, such as:
  - Age vs Cholesterol
  - Resting BP vs Max Heart Rate
  - ST Depression vs Number of Vessels
- For each pair:
  - Subset data to 2D
  - Train logistic regression model
  - Plot decision boundary and labeled scatter plot
- Discuss linear separability and non-linearity
- Include markdown insights (e.g., “Clear divide at cholesterol > 250”)

---

### **Step 4: Logistic Regression with Regularization**
- Add **L2 regularization** to:
  - Cost function: λ/(2m)‖w‖²
  - Gradients: dw += (λ/m)w
- Tune λ values: `[0, 0.001, 0.01, 0.1, 1]`
- Retrain:
  - Full model
  - Selected feature pairs
- Compare:
  - Cost curves (regularized vs unregularized)
  - Decision boundaries
  - Metrics and weight norms
- Report results in tables and plots
- Example conclusion:
  > “Optimal λ = 0.01 improved F1-score by ~5% while reducing model complexity.”

---

### **Step 5: Explore Deployment in Amazon SageMaker**
- Export best model parameters (`w` and `b`) as NumPy arrays
- Use **Amazon SageMaker (Free Tier / Studio)**:
  - Create notebook instance
  - Upload and execute training notebook
  - Build a simple inference script
- Deploy a real-time endpoint
- Test with sample input:
  - Example: Age = 60, Cholesterol = 300
  - Output: Probability = 0.68 (high risk)
- Discuss deployment benefits and latency

---

## Deployment Evidence (Amazon SageMaker)





