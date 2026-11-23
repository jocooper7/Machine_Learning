
# CS 570 Final Project
# Machine Learning Algorithm and Neural Network Implementations
# Author: Jo Cooper

# imports

import numpy as np
import pandas as pd

# import libraries for visualization from matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn libraries for preprocessing
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# sklearn libraries for models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# sklearn libraries for model selection and metrics
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, LearningCurveDisplay
from sklearn.metrics import accuracy_score, f1_score, make_scorer, cohen_kappa_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# neural network dataset for part 2
from sklearn.datasets import load_digits


# flags to show preprocessing and testing
show_preprocessing_knn = False
ML_testing_knn = False

show_preprocessing_multi = False
ML_testing_multi = False

# loading part one dataset into dataframe
df = pd.read_csv("ProjectPartOneDataSet.csv")
df = df[['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'feat6', 'class']]

# loading part two dataset into dataframe
digits = load_digits(as_frame = True)
digit_df = digits.frame
data, target = digits.data, digits.target

## Part 1 Data Preprocessing and Exploration
if show_preprocessing_knn:
    print("\nDataset Info and Description:")
    # shows no entries are null, so imputation is not needed for this data
    print(df.info(), "\n")
    # shows number of unique entries in each column as well as the top entry
    print(df.describe(), "\n")

    print("\nFirst 5 rows of the dataset:")
    print(df.head(5), "\n")

    # printing unique values
    for col in df.columns:
        print(f"Unique values in column '{col}':")
        print("-", end = " ")
        print(df[col].unique())

# performing ordinal encoding based on unique values shown from above
# encoding categorical variables
cols = list(df.columns)
values = [       
    ["low", "med", "high", "vhigh"], # feat1
    ["low", "med", "high", "vhigh"], # feat2
    ["2", "3", "4", "5more"],        # feat3
    ["2", "4", "more"],              # feat4
    ["small", "med", "big"],         # feat5
    ["low", "med", "high"],          # feat6
    ["unacc", "acc", "good", "vgood"]        # class
]

encoder = OrdinalEncoder(categories = values)
encoded_df = pd.DataFrame(encoder.fit_transform(df[cols]), columns = cols)

## Part 1 Preprocessing Continued
if show_preprocessing_knn:
    print("\nEncoded dataframe:")
    print(encoded_df.head(5))

    # creation of encoded data correlation heatmap
    corr_matrix = encoded_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr_matrix, mask = mask, annot = True, cmap = "viridis", fmt = ".2f")
    plt.title("Correlation Matrix of Encoded Dataset")
    plt.tight_layout()
    # plt.savefig("correlation_encoded.png") # can save if needed
    plt.show()

    # creation of distribution histograms
    encoded_df.hist(bins = 30, figsize = (12, 10))
    # plt.savefig("encoded_hist.png") # can save if needed
    plt.show()

## Implementation of Traditional Machine Learning Model
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_df)
scaled_df = pd.DataFrame(scaled_data, columns = encoded_df.columns)
knn_X = scaled_df.drop(columns = "class")
knn_y = df["class"]

knn_train, knn_test, knn_ytrain, knn_ytest = train_test_split(knn_X, knn_y, test_size = 0.2, random_state = 42)

# KNN model hyperparameter testing
if ML_testing_knn:
    knnClass = KNeighborsClassifier()  
    knnClass.fit(knn_train, knn_ytrain)
    print(f"K-nearest Neighbors Score: {knnClass.score(knn_test, knn_ytest)}")

    discrim = LinearDiscriminantAnalysis()
    discrim.fit(knn_train, knn_ytrain)
    print(f"Discriminant Analysis Score: {discrim.score(knn_test, knn_ytest)}")

    naive = GaussianNB()
    naive.fit(knn_train, knn_ytrain)
    print(f"Naive Bayes Score: {naive.score(knn_test, knn_ytest)}")

    # k-fold cross validation for knn
    knnClass = KNeighborsClassifier(n_neighbors = 5) # using default minkowski distance
    cv_results = cross_validate(knnClass, knn_X, knn_y, cv = 5)
    print(f"\nKNN CV scores: {cv_results['test_score']}")
    print(f"KNN CV mean score: {cv_results['test_score'].mean()}\n")

    # full grid search for knn
    pipe = Pipeline([("knn", KNeighborsClassifier())])
    param_grid = {
        "knn__n_neighbors": range(1, 31),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "cityblock", "minkowski"],
    }
    scores = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score),
    }
    knn_grid = GridSearchCV(estimator = pipe, param_grid = param_grid, cv = 5, scoring = scores, refit = "f1",) # refit by f1 given difference in recall and other metrics
    knn_grid.fit(knn_train, knn_ytrain)
    print(f"Best KNN hyperparameters: {knn_grid.best_params_}")
    print(f"Best KNN score: {knn_grid.best_score_}")

    best_mlmodel = knn_grid.best_estimator_
    y_pred = best_mlmodel.predict(knn_test)
    print("\nAccuracy  :", accuracy_score(knn_ytest, y_pred))                   # proportion of correct predictions
    print("Precision :", precision_score(knn_ytest, y_pred, average = "macro")) # proportion of correct positive predictions
    print("Recall    :", recall_score(knn_ytest, y_pred, average = "macro"))    # proportion of correctly predicted positive instances
    print("F1        :", f1_score(knn_ytest, y_pred, average = "macro"))        # harmonic mean of precision and recall
    print("Kappa     :", cohen_kappa_score(knn_ytest, y_pred))                  # compares observed and expected accuracy

    ConfusionMatrixDisplay.from_predictions(knn_ytest, y_pred, cmap = 'viridis')
    plt.show()

    LearningCurveDisplay.from_estimator(
        knn_grid.best_estimator_, knn_train, knn_ytrain, cv = 5,
        scoring = "accuracy", train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
    )
    plt.show()
    LearningCurveDisplay.from_estimator(
        knn_grid.best_estimator_, knn_train, knn_ytrain, cv = 5,
        scoring = "recall_macro", train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
    )
    plt.show()

knnClass = KNeighborsClassifier(n_neighbors = 8, metric = 'cityblock', weights = 'distance')
knnClass.fit(knn_train, knn_ytrain)
print(f"\nFinal KNN Accuracy Score: {knnClass.score(knn_test, knn_ytest)}\n")


## Multilayer Processor Progamming

# Part 2 Data Preprocessing and Exploration
if show_preprocessing_multi:
    # no null values are present in the dataset
    digit_df.info()
    print()
    digit_df.describe()

    # creation of pixel data correlation heatmap
    digit_corr = digit_df.drop(columns = "target").corr()
    digit_mask = np.triu(np.ones_like(digit_corr, dtype = bool))
    plt.figure(figsize = (10, 8))
    sns.heatmap(digit_corr, cmap = "viridis", center = 0, square = True)
    plt.title("Pixel Value Correlation Matrix")
    plt.tight_layout()
    # plt.savefig("correlation_digits.png") # can save if needed
    plt.show()

    # analyzing data of empty rows/columns in correlation matrix
    print('\n', data[["pixel_0_0", "pixel_4_0", "pixel_4_7"]].nunique())
    print(data[["pixel_0_0", "pixel_4_0", "pixel_4_7"]].describe())

    # distribution of all pixel data combined
    plt.figure(figsize = (6, 4))
    plt.hist(data.values.flatten(), bins = 17)
    plt.title("Distribution of pixel values")
    plt.xlabel("Pixel value (0-16)")
    plt.show()

    # distribution of target outputs
    plt.figure(figsize = (6, 4))
    sns.countplot(x = target)
    plt.title("Distribution of digit classes")
    plt.show()

dig_X = digit_df.drop(columns = "target")
scaler = StandardScaler()
scaled_digits = scaler.fit_transform(dig_X)
scaled_dig_df = pd.DataFrame(scaled_digits, columns = dig_X.columns)
dig_X = scaled_dig_df
dig_y = digit_df["target"]

dig_train, dig_test, dig_ytrain, dig_ytest = train_test_split(dig_X, dig_y, test_size = 0.2, random_state = 42)

# MLP classifier testing
if ML_testing_multi:
    mp_class = MLPClassifier(hidden_layer_sizes = (50, 50, ))
    dig_cv_res = cross_validate(mp_class, dig_X, dig_y, cv = 5)
    print(f"Multilayer CV scores: {dig_cv_res['test_score']}")
    print(f"Mean CV score: {dig_cv_res['test_score'].mean()}")


    pipe = Pipeline([("mlp", MLPClassifier(random_state = 42)),])

    mp_param_grid = {  
        "mlp__solver": ["sgd", "adam", "lbfgs"],
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__max_iter": [500, 1000],
        "mlp__batch_size": [50, 100],
        }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score),
    }

    dig_grid = GridSearchCV(estimator = pipe, param_grid = mp_param_grid, cv = 5, scoring = scoring, refit = "f1", n_jobs = -1) # refit with f1 for consistency in program
    dig_grid.fit(dig_train, dig_ytrain)

    print(f"\nBest MLP hyperparameters: {dig_grid.best_params_}")
    print(f"Best MLP score: {dig_grid.best_score_}")

    best_MLP = dig_grid.best_estimator_
    mlp_pred = best_MLP.predict(dig_test)
    print(f"\nMLP Accuracy: {accuracy_score(dig_ytest, mlp_pred)}")                  # proportion of correct predictions
    print(f"MLP Precision: {precision_score(dig_ytest, mlp_pred, average='macro')}") # proportion of correct positive predictions
    print(f"MLP Recall: {recall_score(dig_ytest, mlp_pred, average='macro')}")       # proportion of correctly predicted positive instances
    print(f"MLP F1: {f1_score(dig_ytest, mlp_pred, average='macro')}")               # harmonic mean of precision and recall
    print(f"MLP Kappa: {cohen_kappa_score(dig_ytest, mlp_pred)}")                    # compares observed and expected accuracy

    ConfusionMatrixDisplay.from_predictions(dig_ytest, mlp_pred, cmap = 'viridis')
    plt.show()

    LearningCurveDisplay.from_estimator(
        best_MLP, dig_train, dig_ytrain, cv = 5,
        scoring = "accuracy", train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
    )
    plt.show()

    LearningCurveDisplay.from_estimator(
            best_MLP, dig_train, dig_ytrain, cv = 5,
            scoring = "recall_macro", train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
        )
    plt.show()


best_MLP = MLPClassifier(activation = 'relu', batch_size = 50, hidden_layer_sizes = (100,), max_iter = 500, solver = 'adam', random_state = 42)
best_MLP.fit(dig_train, dig_ytrain)
mlp_pred = best_MLP.predict(dig_test)
print(f"Final MLP Accuracy Score: {accuracy_score(dig_ytest, mlp_pred)}")

