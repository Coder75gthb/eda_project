# IMPORTING DATASETS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df=pd.read_csv("student_habits_performance.csv")

## DOING ANALYSIS OF DATA
# print(df.info())

# ALL THE DATA TYPES ARE FINE,PARENTAL_EDUCATION_LEVEL CONTAINS NULL VALUES 
print(df.shape)

# 1000 ROWS AND 16 COLUMNS
print(df.isnull().sum())
#91 NULL VALUES IN P_E_L
print(df.parental_education_level.mode())
print(df.parental_education_level.value_counts())
df1=df
# HIGH SCHOOL IS THE MODE FOR P_E_L
df1.parental_education_level=df.parental_education_level.fillna("High School")
##print(df1.info())

##NULL VALUES ARE TREATED

##PLOTTNG THE BOXPLOTS JUST TO VISUALLY CHECK THE OUTLIERS 
fig,axes=plt.subplots(3,3)
plt.tight_layout(pad=2)
axes[0,0].boxplot(x=df1.age)
axes[0,1].boxplot(df1.study_hours_per_day)
axes[0,2].boxplot(df1.social_media_hours)
axes[1,0].boxplot(df1.netflix_hours)
axes[1,1].boxplot(df1.attendance_percentage)
axes[1,2].boxplot(df1.sleep_hours)
axes[2,0].boxplot(df1.exercise_frequency)
axes[2,1].boxplot(df1.mental_health_rating)
axes[2,2].boxplot(df1.exam_score)
plt.show()

for col in df1.select_dtypes(include=['float64', 'int64']):
    
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    is_outlier = (df1[col] < lower) | (df1[col] > upper)

    # Replace outliers with mean
    
    df1.loc[is_outlier, col] = df1[col].mean()

# CATEGORICAL DATA-gender,part time job,diet quality,parental education level,internet_quality,ep

print(df1.gender.value_counts())
print(df1.part_time_job.value_counts())
print(df1.diet_quality.value_counts())
print(df1.exercise_frequency.value_counts())
print(df1.parental_education_level.value_counts())
print(df1.internet_quality.value_counts())
print(df1.extracurricular_participation.value_counts())

## PERFORMING DV
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
plt.tight_layout(pad=2)

sns.countplot(x='gender', data=df1, ax=axes[0, 0])
sns.countplot(x='part_time_job', data=df1, ax=axes[0, 1])
sns.countplot(x='diet_quality', data=df1, ax=axes[1, 0])
sns.countplot(x='exercise_frequency', data=df1, ax=axes[1, 1])
sns.countplot(x='parental_education_level', data=df1, ax=axes[2, 0])
sns.countplot(x='internet_quality', data=df1, ax=axes[2, 1])
plt.show()
sns.countplot(df1.extracurricular_participation)
plt.show()

## USING LABEL ENCODING TO MAP CATEGORICAL VARIABLES TO NUMERIC TYPE IN ORDER TO PLOT HEATMAP

df1_numeric = df1.select_dtypes(include=['number'])
df1_numeric['gender'] = df1['gender'].map({'Male': 0, 'Female': 1})
df1_numeric['part_time_job']=df1['part_time_job'].map({'No':0,'Yes':1})
# print(df1_numeric['part_time_job'])
plt.figure(figsize=(10, 10))
corr = df1_numeric.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

## MULTIPLE CATEGORY VARIABLES-diet_quality,parental_education_level,internet_quality
numeric_cols = df.select_dtypes(include='number').columns
grouped1 = df.groupby('parental_education_level')[numeric_cols].mean()
grouped2=df.groupby('diet_quality')[numeric_cols].mean()
grouped3=df.groupby('internet_quality')[numeric_cols].mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))  

## PLOTTING HEATMAPS FOR EACH GROUPED DATAFRAME
sns.heatmap(grouped1, ax=axes[0], annot=True, cmap='YlGnBu')
axes[0].set_title("Heatmap 1")

sns.heatmap(grouped2, ax=axes[1], annot=True, cmap='coolwarm')
axes[1].set_title("Heatmap 2")

sns.heatmap(grouped3, ax=axes[2], annot=True, cmap='coolwarm')
axes[2].set_title("Heatmap 3")

plt.tight_layout()
plt.show()

# ONE HOT ENCODING FOR-DIET QUALITY,PARENTAL EDUCATION LEVEL,INTERNET_QUALITY,EP
categorical_cols = df1.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)

##SPLITTING THE DATASET INTO TRAINING AND TESTING SETS

X = df_encoded.drop('exam_score', axis=1)
y = df_encoded['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## TRAINING A LINEAR REGRESSION MODEL

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## EVALUATING THE MODEL

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")





