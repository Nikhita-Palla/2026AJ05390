# Online Shopper Purchase Intention — ML Classification (Assignment 2)

**Live app:** `<add your Streamlit Community Cloud URL here>`

## a. Problem statement

Given the browsing behaviour of a visitor during a single session on an e-commerce website, predict whether the session will end in a **purchase** (`Revenue = 1`) or not (`Revenue = 0`). This is a **binary classification** problem. Five classifiers are trained on the same data, compared on six evaluation metrics, and served in an interactive Streamlit app that evaluates the saved models on uploaded test data without retraining.

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

**Preprocessing** (part of every saved pipeline, so the app takes raw columns):
- `Weekend` and `Revenue` converted to 0/1; duplicate rows dropped.
- Numeric features standardised with `StandardScaler`.
- `Month` and `VisitorType` one-hot encoded (`handle_unknown="ignore"`).
- Stratified 80/20 train/test split (`random_state=42`). The 20 % test split (2,441 rows) is saved as `test_data.csv`.

## c. GitHub repository link

`<add your GitHub repository URL here>`

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

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong ranking ability (AUC 0.90, second best) and high precision (0.77), but low recall (0.42): at the default 0.5 threshold it misses more than half of the buyers because of the 84/16 class imbalance. A good, interpretable baseline; lowering the threshold would trade precision for recall. |
| Decision Tree | Best on almost every metric — accuracy 0.90, AUC 0.93, F1 0.66, MCC 0.60 — with the highest recall among the accurate models (0.61). Axis-aligned splits capture the threshold-like effect of `PageValues` (sessions with PageValue > 0 are far more likely to buy) better than the linear/distance-based models. Depth and leaf-size limits keep it from overfitting. |
| KNN | Weakest of the "high-accuracy" models (MCC 0.46, recall 0.35). With many features of mixed importance, the distance metric is dominated by less useful features, and the majority class dominates neighbourhoods. It is also the largest saved model because it stores the whole training set. |
| Naive Bayes | Lowest accuracy (0.71) and MCC (0.38) but the highest recall (0.81): it flags most buyers at the cost of many false positives (precision 0.33). The independence and Gaussian assumptions are violated (page counts and durations are strongly correlated and heavily skewed), which distorts its probability estimates. |
| Support Vector Machine | Second-best accuracy (0.89), F1 (0.59) and MCC (0.55). The RBF kernel captures non-linear structure better than logistic regression, but recall (0.48) is still limited by class imbalance, and it is the slowest model to train. |
| **Overall winner** | **Decision Tree** — the highest Accuracy, AUC, F1 and MCC. MCC and F1 are the most informative metrics for this imbalanced dataset, and the Decision Tree leads both by a clear margin. |

## Streamlit app features

- **CSV upload** of test data, or the bundled sample `test_data.csv` (a download button is provided).
- **Model selection dropdown** for the five saved models, with a one-line description of each.
- **Model results** tab: a plain-English summary, the six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) with explanations, a labelled **confusion matrix**, outcome counts (buyers caught/missed, false alarms) and the **classification report**.
- **Predictions** tab: row-by-row actual vs. predicted outcome, purchase probability bar and ✅/❌ result. It can be filtered (correct, wrong, missed buyers…) and downloaded as CSV.
- **Compare all models** tab: metric table with the best value per metric highlighted, the overall winner, a per-metric bar chart and ROC curves.
- **Data** tab: preview of the test data and a guide to each column.

The app loads the saved `.joblib` pipelines and **only predicts** on the uploaded data — no retraining. The uploaded CSV must contain all 17 feature columns and the `Revenue` target column (`TRUE/FALSE` or `1/0`).

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
