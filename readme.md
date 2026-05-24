# Student Habits and Performance: EDA + Linear Regression 📊

> Exploring how student lifestyle choices — sleep, diet, social media, and more — impact academic performance.

This project performs end-to-end data analysis and machine learning on a real-world student dataset. It covers data cleaning, visual EDA, feature engineering, and a Linear Regression model that achieves an **R² score of 0.818** in predicting exam scores.

---

## Dataset

| Property | Details |
|---|---|
| File | `student_habits_performance.csv` |
| Rows | 1000 |
| Columns | 16 |
| Target Variable | `exam_score` |

---

## Project Workflow

### 1. Data Loading
- Loaded the dataset using `pandas`

### 2. Data Cleaning
- Imputed missing values in `parental_education_level` with the mode (`'High School'`)
- Detected and treated outliers in numeric columns using the **IQR method**, replacing them with column means (validated using boxplots)

### 3. Exploratory Data Analysis (EDA)
- Visualized categorical variables (gender, job status, diet quality, etc.) using count plots
- Plotted a **correlation heatmap** to study relationships among numerical features
- Generated group-wise heatmaps segmented by `parental_education_level`, `diet_quality`, and `internet_quality`

### 4. Feature Engineering & Encoding
- **Label Encoding** — applied to binary categorical features (e.g. gender) where ordinal significance is not a concern
- **One-Hot Encoding** — applied to multi-class categorical features with `drop_first=True` to avoid the dummy variable trap

### 5. Model Building
- Algorithm: **Linear Regression** via scikit-learn
- Train/test split: **80% training, 20% testing**

### 6. Results

| Metric | Score |
|---|---|
| R² Score | **0.818** |
| MAE | 4.55 |
| RMSE | 6.51 |
| MSE | 42.43 |

An R² of **0.818** means the model explains ~82% of the variance in student exam scores — strong performance for a linear model on behavioral data.

---

## Key Findings

- Study habits and sleep have the strongest positive correlation with exam scores
- High social media usage shows a notable negative relationship with performance
- Diet quality and internet quality create meaningful group-level differences in score distributions

---

## Tech Stack

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn`

