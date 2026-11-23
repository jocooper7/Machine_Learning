# Module 5 - Student Starter Code 

# This code is provided as a starting point for the assignment.
# You are expected to complete the code and implement the required functionality. 
# TODO: Import libraries for whatever you need to complete the assignment (the libraries below are for reference and are incomplete).
import numpy as np
import pandas as pd

# Import libraries for visualization from matplotlib and seaborn and inspection from sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Import libraries for preprocessing from sklearn
from sklearn.preprocessing import OrdinalEncoder

# Import libraries for imputation from sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

# Import libraries for model model selection and metrics from sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Import libraries for regression from sklearn
from sklearn.linear_model import LinearRegression

# Function to test dataset on a linear regression model
# Use this function to compare how different imputation methods affect the model performance
def get_score(df, imputer=None): 

    # Define features (X) and target (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    regressor = LinearRegression()
    if imputer is not None:
        estimator = make_pipeline(imputer, regressor)
    else:
        estimator = regressor
    scores = cross_val_score(
        estimator, X, y, scoring="neg_mean_squared_error", cv=4
    )
    return scores.mean(), scores.std()

#----------------------------------------------------------------
# Step 1: Load the CSV file into a pandas DataFrame
#----------------------------------------------------------------
df = pd.read_csv("Module5Dataset.csv")  
df = df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Class']]

#----------------------------------------------------------------
# Step 2: Data Exploration and Visualization
#         Does the data contain NAN or nan values (implying incomplete data)?
#         What type of data is in the dataset?
#----------------------------------------------------------------
print("-" * 50)
print("\nStep 2: Data Exploration and Visualization")
print("\nDataset Description:")
print("Row count: ", len(df))
print(df.describe(), "\n")

for col in df.columns:
    print(f"Unique values in column '{col}':")
    print("-", end = " ")
    print(df[col].unique())

print("\nFirst 4 rows of the dataset:")
print(df.head(4))
print("-" * 50)

#----------------------------------------------------------------
# Step 3: Does the dataset work on the linear regression model before preprocessing?
#         What type of data does the sklearn linear regression model accept?
#----------------------------------------------------------------
print("-" * 50)
print("\nStep 3: Checking if the dataset works with a linear regression model before preprocessing")

print("\nAttempting to run step 3's templated code:")
x_labels = []
mses = np.zeros(5)
stds = np.zeros(5)

try:
    mses[0], stds[0] = get_score(df) # Does it work?
except Exception as e:
    print("Error occurred while attempting code.")
finally:
    x_labels.append("RAW Dataset")

print("\nAttempting to fit Linear Regression model on raw dataset (before preprocessing):")
lrModel = LinearRegression()
try:
    X = df.drop("Class", axis = 1)
    y = df["Class"]
    lrModel.fit(X, y)

    print("\nLinear Regression model coefficients:")
    print(lrModel.coef_)
    print("Linear Regression model intercept:")
    print(lrModel.intercept_)

except Exception as e:
    print("\nError occurred while fitting Linear Regression model on raw dataset:")
    print(e)

#----------------------------------------------------------------
# Step 4: Transform the dataset
#         Note: It is possible to impute missing values in the dataset before transforming it.
#               or even at the same time.
#----------------------------------------------------------------
print("-" * 50)
print("\nStep 4: Transforming the dataset by encoding categorical variables")

# Encoding categorical variables
# Entries appear to be ordinal in nature, so encoding takes place using OrdinalEncoder
cols = list(df.columns)
values = [       
    ["low", "med", "high", "vhigh", np.nan], # Feature1
    ["low", "med", "high", "vhigh", np.nan], # Feature2
    ["2", "3", "4", "5more", np.nan],        # Feature3
    ["2", "4", "more", np.nan],              # Feature4
    ["small", "med", "big", np.nan],         # Feature5
    ["low", "med", "high", np.nan],          # Feature6
    ["unacc", "acc", "good", "vgood"]        # Class
]

encoder = OrdinalEncoder(categories = values)
encoded_df = pd.DataFrame(encoder.fit_transform(df[cols]), columns = cols)

# Creation of correlation matrix
corr_matrix = encoded_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask = mask, annot = True, cmap = "viridis", fmt = ".2f")
plt.title("Correlation Matrix of Encoded Dataset")
plt.tight_layout()
# plt.savefig("correlation_encoded.png") # can save if needed
plt.show()
print("Matrix shows that features are uncorrelated or weakly correlated")

print("\nAttempting to fit Linear Regression model on encoded dataset (before handling NaN values):")
lrModel = LinearRegression()
try:
    X = encoded_df.drop("Class", axis = 1)
    y = encoded_df["Class"]
    lrModel.fit(X, y)

    print("\nLinear Regression model coefficients after encoding:")
    print(lrModel.coef_)
    print("Linear Regression model intercept after encoding:")
    print(lrModel.intercept_)

except Exception as e:
    print("\nError occurred while fitting Linear Regression model on encoded dataset:")
    print(e)

#----------------------------------------------------------------
# Step 5: Impute missing values in the dataset (replace NAN or nan values)
#         Experiment with the following imputers: 
#         - SimpleImputer with strategy='constant' and fill_value=0
#         - SimpleImputer with strategy='mean'
#         - KNNImputer
#         - IterativeImputer
#----------------------------------------------------------------
print("-" * 50)
print("\nStep 5: Imputing missing values in the encoded dataset")

imputer_zero = SimpleImputer(strategy="constant", fill_value=0)
mses[1], stds[1] = get_score(encoded_df, imputer_zero)
x_labels.append("Zero Imputation")

imputer_mean = SimpleImputer(strategy="mean")
mses[2], stds[2] = get_score(encoded_df, imputer_mean)
x_labels.append("Mean Imputation")

imputer_knn = KNNImputer()
mses[3], stds[3] = get_score(encoded_df, imputer_knn)
x_labels.append("KNN Imputation")

imputer_iter = IterativeImputer(max_iter=10, random_state=42)
mses[4], stds[4] = get_score(encoded_df, imputer_iter)
x_labels.append("Iterative Imputation")

#----------------------------------------------------------------
# Step 6: TODO: Collect data metrics and visualization of:
#         Dataset after transforming
#         Dataset after imputing missing values
#         sns.heatmap(?????????.corr(), annot=True, cmap="viridis", fmt=".2f")
#--------------------------------------------------------------------

print(x_labels)
print("Negative Mean Squared Errors (MSEs):", mses)
print("Standard Deviations (STDs):", stds)

# Plot the results
plt.figure(figsize=(10, 8))
plt.bar(x_labels, stds, yerr=stds, capsize=5)
plt.xlabel("Imputation Method")
plt.ylabel("Negative Mean Squared Error")
plt.title("Comparison of Imputation Methods")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig("imputation_comparison.png")
plt.show()

# Create a new DataFrame with the imputed values using the mean imputer
imputed_df = pd.DataFrame(imputer_mean.fit_transform(encoded_df), columns=encoded_df.columns)

# Information on dataset after imputing missing values
print("\nDataset Description after imputing missing values:")
print("Row count: ", len(imputed_df))
print(imputed_df.describe(), "\n")

for col in imputed_df.columns:
    print(f"Unique values in column '{col}':")
    print("-", end = " ")
    print(imputed_df[col].unique())

# Visualize the correlation matrix of the imputed dataset
corr_matrix = imputed_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask = mask, annot = True, cmap = "viridis", fmt = ".2f")
plt.title("Correlation Matrix of Imputed Dataset")
plt.tight_layout()
# plt.savefig("correlation_imputed.png")
plt.show()

# Dataset is encoded and missing values have been imputed
print("\nTop 5 rows of the imputed dataset:")
print(imputed_df.head())
