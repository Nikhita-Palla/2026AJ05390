# Online Shopper Purchase Intention — ML Classification (Assignment 2)

**Live app:** https://shopper-intent-ml-2026aj05390.streamlit.app/

## a. Problem statement

The goal is to predict whether a visitor's session on an online shopping website ends in a purchase (`Revenue = 1`) or not (`Revenue = 0`), based on how they browsed the site. It is a binary classification problem. I trained 5 classifiers on the same data, compared them using 6 metrics, and built a Streamlit app where the saved models can be tested on uploaded test data.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions (12,205 after removing 125 duplicate rows) |
| Features | 17 input features + 1 target |
| Target | `Revenue` (TRUE/FALSE → 1/0) |
| Class balance | 84.4 % no purchase, 15.6 % purchase (imbalanced) |
| Missing values | None |

**Features**

| Feature | Type | Description |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric | Number of pages of each type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric | Time (s) spent on each page type |
| BounceRates | numeric | Avg. bounce rate of pages visited (Google Analytics) |
| ExitRates | numeric | Avg. exit rate of pages visited |
| PageValues | numeric | Avg. page value of pages visited before a transaction |
| SpecialDay | numeric | Closeness of the visit date to a special day (e.g. Valentine's Day) |
| Month | categorical | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | integer-coded | Visitor's OS, browser, region and traffic source |
| VisitorType | categorical | Returning_Visitor / New_Visitor / Other |
| Weekend | boolean | Whether the visit was on a weekend |

**Preprocessing** (in `train_models.py`)

1. Removed 125 duplicate rows, so 12,205 rows were left. There were no missing values.
2. Changed `Weekend` and `Revenue` from TRUE/FALSE to 1/0.
3. Split the data 80/20 with stratify, so both parts have about 15.6 % buyers. This gave 9,764 rows for training and 2,441 rows for testing. I saved the test part as `test_data.csv`.
4. Scaled the 15 numeric columns with `StandardScaler`, fitted only on the training data. KNN, SVM and Logistic Regression need this because they depend on the scale of the features.
5. One-hot encoded `Month` and `VisitorType`.
6. Saved the preprocessing and the model together in one pipeline for each model. Because of this, the app can take the raw CSV and apply the same steps before predicting.

## c. GitHub repository link

https://github.com/Nikhita-Palla/2026AJ05390

```
project-folder/
├── app.py                  # Streamlit app
├── train_models.py         # Trains & saves all 5 models, writes test_data.csv
├── metrics_utils.py        # Shared metric computation (training + app)
├── requirements.txt
├── README.md
├── test_data.csv           # Held-out test data (upload this in the app)
├── data/
│   └── online_shoppers_intention.csv   # Original UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.json        # Test-set metrics from training
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)` |
| KNN | `KNeighborsClassifier(n_neighbors=15, weights="distance")` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | `SVC(kernel="rbf", C=1.0)` wrapped in `CalibratedClassifierCV` for probability scores (AUC) |

### Comparison table (test set, 2,441 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8894 | 0.9022 | 0.7718 | 0.4162 | 0.5408 | 0.5142 |
| Decision Tree | 0.9005 | 0.9276 | 0.7125 | 0.6099 | 0.6573 | 0.6019 |
| KNN | 0.8791 | 0.8547 | 0.7377 | 0.3534 | 0.4779 | 0.4554 |
| Naive Bayes | 0.7108 | 0.8268 | 0.3284 | 0.8115 | 0.4676 | 0.3757 |
| Support Vector Machine | 0.8939 | 0.8798 | 0.7531 | 0.4791 | 0.5856 | 0.5460 |

### Observations

**Ranking by MCC** (fairest single metric here, as only 16 % of sessions are purchases):
Decision Tree (0.60) > SVM (0.55) > Logistic Regression (0.51) > KNN (0.46) > Naive Bayes (0.38)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Reliable baseline, but too cautious.**<br>• High precision (0.77) and AUC 0.90 (2nd best)<br>• Low recall (0.42): misses over half of the buyers<br>• Cause: default 0.5 threshold with only 16 % buyers |
| Decision Tree | **Best overall.**<br>• Highest Accuracy 0.90, AUC 0.93, F1 0.66, MCC 0.60<br>• Catches 61 % of buyers<br>• Splits mainly on `PageValues` (75 % of its decisions); depth limit prevents overfitting |
| KNN | **Weakest of the high-accuracy models.**<br>• Accuracy 0.88, but lowest recall (0.35) and MCC 0.46<br>• Neighbourhoods are dominated by non-buyers<br>• Weighs all features equally, diluting strong signals like `PageValues` |
| Naive Bayes | **Catches buyers, but too many false alarms.**<br>• Highest recall (0.81)<br>• Lowest precision (0.33), accuracy (0.71) and MCC (0.38)<br>• Independence and normality assumptions break: features are correlated and skewed |
| Support Vector Machine | **Strong runner-up.**<br>• 2nd-best Accuracy 0.89, F1 0.59, MCC 0.55<br>• RBF kernel captures non-linear patterns<br>• Recall (0.48) held back by class imbalance; slowest to train |
| **Overall winner** | **Decision Tree.**<br>• Leads on Accuracy, AUC, F1 and MCC<br>• F1 and MCC matter most with imbalanced classes; MCC 0.60 vs 0.55 for SVM |

## Streamlit app features

The app has 3 pages in the sidebar. Below them there is an option to choose the test data.

- Test data: upload a CSV, or use the sample `test_data.csv` (it can also be downloaded from the app).
- Overview: short description of the app and the dataset, and a summary of the loaded data.
- Evaluate a model: model dropdown, the 6 metrics, confusion matrix, classification report and a table of predictions for each row.
- Compare all models: metrics of all 5 models in one table, the best model, a bar chart and ROC curves.

The app does not train anything. It loads the saved `.joblib` models and only predicts on the uploaded data. The CSV must have all 17 feature columns and the `Revenue` column.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional: re-creates model/*.joblib and test_data.csv
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud → sign in with GitHub → **New app**.
3. Select the repository, branch `main`, main file `app.py`.
4. In **Advanced settings**, choose the same Python version used for training (3.14) so the saved models load with the pinned scikit-learn version.
5. Click **Deploy**.
