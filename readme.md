# Student Habits and Performance: EDA + Linear Regression

This project explores how various student habits — like sleep, social media usage, diet, and more affect their academic performance. The data is cleaned, analyzed visually, and modeled using Linear Regression to predict exam scores.

---

## Dataset

-File Name: `student_habits_performance.csv`
-Rows: 1000
-Columns:16
-Target Variable: `exam_score`

---

## Project Workflow

### 1. Data Loading
- Loaded the dataset using `pandas`

### 2. Data Cleaning
- Treated missing values in `parental_education_level` by replacing them with 'High School'(mode).
- Treated outliers in numeric columns using the IQR method, replacing them with the mean(checked using boxplots).

### 3. Exploratory Data Analysis (EDA)
- Visualized categorical variables like gender, job status, diet quality, etc. using count plots.
- Plotted a correlation heatmap to study relationships among numerical features.
- Generated group-wise heatmaps based on "parental_education_level", "diet_quality", and "internet_quality".

### 4. Data Encoding
- Used Label Encoding on Categorical features which contain only 2 options and wont get affected by the numerical significance(eg:gender)
- Used One-Hot Encoding on categorical features having more than 2 options (drop_first=True to avoid dummy variable trap).

### 5. Model Building
- Applied Linear Regression using scikit-learn.
- Split data: 80% for training, 20% for testing.

### 6. Evaluation Metrics
Evaluated the model using:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

