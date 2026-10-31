"""
Streamlit app: evaluate pre-trained classifiers on uploaded test data for the
UCI Online Shoppers Purchasing Intention dataset.

Run locally:  python -m streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from metrics_utils import compute_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

TARGET = "Revenue"
FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
    "Month", "VisitorType",
]
# Columns shown next to each prediction so a row is recognisable at a glance
KEY_COLUMNS = ["Month", "VisitorType", "ProductRelated", "PageValues", "BounceRates", "ExitRates"]

MODELS = {
    "Logistic Regression": ("logistic_regression.joblib",
                            "Linear model that weighs each feature to estimate the chance of a purchase."),
    "Decision Tree": ("decision_tree.joblib",
                      "Learns a set of yes/no rules, e.g. “is PageValues above X?”."),
    "KNN": ("knn.joblib",
            "Looks at the 15 most similar past sessions and follows the majority."),
    "Naive Bayes": ("naive_bayes.joblib",
                    "Probabilistic model that assumes features are independent of each other."),
    "Support Vector Machine": ("support_vector_machine.joblib",
                               "Finds the boundary that best separates buyers from non-buyers."),
}

METRIC_HELP = {
    "Accuracy": "Share of all sessions the model classified correctly.",
    "AUC": "How well the model ranks buyers above non-buyers (0.5 = random guess, 1 = perfect).",
    "Precision": "When the model says “will buy”, how often is it right?",
    "Recall": "Of all real buyers, how many did the model catch?",
    "F1": "Balance of precision and recall in one number.",
    "MCC": "Correlation between predictions and truth (−1 to 1). Fair even when classes are imbalanced.",
}

BUY, NO_BUY = "Purchase", "No purchase"
ACCENT = "#7c5cbf"

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.5rem; max-width: 1150px;}
      div[data-testid="stMetric"] {
          border: 1px solid rgba(124, 92, 191, 0.25);
          border-radius: 10px; padding: 10px 14px;
      }
      .lead {font-size: 1.1rem; line-height: 1.6;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODELS[name][0])


def to_binary(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE strings, booleans or 0/1 values to integers 0/1."""
    return (
        series.astype(str).str.strip().str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    )


def prepare(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing these required columns: {', '.join(missing)}")
    df = df.copy()
    df["Weekend"] = to_binary(df["Weekend"])
    df[TARGET] = to_binary(df[TARGET])
    if df[[TARGET, "Weekend"]].isna().any().any():
        raise ValueError("'Revenue' and 'Weekend' must contain only TRUE/FALSE or 0/1 values.")
    return df[FEATURES], df[TARGET].astype(int)


def evaluate(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    model = load_model(name)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {"y_pred": y_pred, "y_prob": y_prob, "metrics": compute_metrics(y, y_pred, y_prob)}


def plot_confusion(y, y_pred):
    cm = confusion_matrix(y, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Purples", cbar=False, annot_kws={"size": 13},
                xticklabels=["Predicted:\nPurchase", "Predicted:\nNo purchase"],
                yticklabels=["Actual:\nPurchase", "Actual:\nNo purchase"], ax=ax)
    ax.tick_params(length=0, labelsize=9)
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def plot_roc(y, results: dict):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {res['metrics']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random guess")
    ax.set_xlabel("False positive rate (non-buyers wrongly flagged)")
    ax.set_ylabel("True positive rate (buyers caught)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def data_status():
    """One-line reminder of which data the page is using; stops the page if none is loaded."""
    if DATA is None:
        st.info("No test data loaded yet. Choose or upload a CSV in the sidebar.", icon=":material/upload_file:")
        st.stop()
    X, y, _, source = DATA
    st.caption(f"Data: **{source}** · {len(X):,} sessions · {int(y.sum()):,} purchases")


# ---------------------------------------------------------------- pages
def overview_page():
    st.title("Will this shopper buy?")
    st.markdown(
        "<p class='lead'>Each row in the test data is <b>one visit to an online store</b>. "
        "Five pre-trained machine-learning models predict whether the visit ends in a "
        "<b>purchase</b>, and this app checks those predictions against what actually happened.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("How to use")
    s1, s2, s3 = st.columns(3)
    s1.markdown("**1. Pick test data**  \nUse the sample data or upload your own CSV in the sidebar.")
    s2.markdown("**2. Evaluate a model**  \nChoose a model to see its metrics, confusion matrix and predictions.")
    s3.markdown("**3. Compare models**  \nSee all five models side by side and which one wins.")

    st.subheader("Loaded data")
    if DATA is None:
        st.info("No test data loaded yet. Choose or upload a CSV in the sidebar.", icon=":material/upload_file:")
    else:
        X, y, raw, source = DATA
        c1, c2, c3 = st.columns(3)
        c1.metric("Shopping sessions", f"{len(X):,}")
        c2.metric("Purchased", f"{int(y.sum()):,}", help="Sessions where Revenue = TRUE")
        c3.metric("Did not purchase", f"{int((y == 0).sum()):,}")
        with st.expander("Preview the data"):
            st.dataframe(raw.head(50), width="stretch")

    with st.expander("About the dataset and columns"):
        st.markdown(
            """
**Source:** UCI Machine Learning Repository — *Online Shoppers Purchasing Intention* (12,330 sessions, 17 features).
The models were trained on 80 % of the data; the sample test data is the remaining 20 %.

- **Administrative / Informational / ProductRelated** — pages of each type visited; `_Duration` = seconds spent.
- **BounceRates / ExitRates** — how often visitors leave from the pages viewed (lower is better).
- **PageValues** — average value of pages visited before a purchase; the strongest signal of buying.
- **SpecialDay** — closeness to a holiday such as Valentine's Day (0–1).
- **Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType** — visit context.
- **Revenue** — the target: did the session end in a purchase?
"""
        )


def evaluate_page():
    st.title("Evaluate a model")
    data_status()
    X, y, raw, _ = DATA

    # Widget state is dropped when leaving a page, so the choice is kept in model_choice
    st.session_state.model_widget = st.session_state.model_choice

    def remember_choice():
        st.session_state.model_choice = st.session_state.model_widget

    col_sel, col_desc = st.columns([1, 2], vertical_alignment="bottom")
    model_name = col_sel.selectbox("Model", list(MODELS), key="model_widget", on_change=remember_choice)
    col_desc.caption(MODELS[model_name][1])

    res = evaluate(model_name, X, y)
    m, y_pred = res["metrics"], res["y_pred"]
    correct = int((y_pred == y).sum())
    caught = int(((y_pred == 1) & (y == 1)).sum())
    missed = int(((y_pred == 0) & (y == 1)).sum())
    false_alarm = int(((y_pred == 1) & (y == 0)).sum())
    true_neg = int(((y_pred == 0) & (y == 0)).sum())

    st.markdown(
        f"<p class='lead'>Got <b>{correct:,} of {len(y):,}</b> sessions right "
        f"(<b>{m['Accuracy']:.1%}</b>) · caught <b>{caught:,} of {int(y.sum()):,}</b> buyers · "
        f"<b>{false_alarm:,}</b> false alarms</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    for col, key in zip(cols, METRIC_HELP):
        col.metric(key, f"{m[key]:.4f}", help=METRIC_HELP[key])

    tab_perf, tab_preds = st.tabs(["Performance", "Predictions"])

    with tab_perf:
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.markdown("##### Confusion matrix")
            st.pyplot(plot_confusion(y, y_pred))
        with right:
            st.markdown("##### What it means")
            o1, o2 = st.columns(2)
            o1.metric("Buyers caught", f"{caught:,}", help="Actually bought, model predicted purchase")
            o2.metric("Buyers missed", f"{missed:,}", help="Actually bought, model predicted no purchase")
            o3, o4 = st.columns(2)
            o3.metric("False alarms", f"{false_alarm:,}", help="Did not buy, but model predicted purchase")
            o4.metric("Non-buyers correct", f"{true_neg:,}", help="Did not buy, model predicted no purchase")

        st.markdown("##### Classification report")
        report = classification_report(
            y, y_pred, labels=[0, 1], target_names=["No purchase", "Purchase"],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                             "f1-score": "{:.4f}", "support": "{:,}"}),
                     width="stretch")

    with tab_preds:
        preds = pd.DataFrame({
            "Actual": pd.Series(y.values).map({1: BUY, 0: NO_BUY}),
            "Predicted": pd.Series(y_pred).map({1: BUY, 0: NO_BUY}),
            "Purchase chance": res["y_prob"],
            "Result": pd.Series(y_pred == y.values).map({True: "✅ Correct", False: "❌ Wrong"}),
        })
        preds = pd.concat([preds, raw[KEY_COLUMNS].reset_index(drop=True)], axis=1)
        preds.index = preds.index + 1

        f_show, f_sort = st.columns([3, 1], vertical_alignment="bottom")
        show = f_show.segmented_control(
            "Show", ["All", "Correct", "Wrong", "Missed buyers"], default="All"
        )
        if f_sort.toggle("Highest chance first"):
            preds = preds.sort_values("Purchase chance", ascending=False)

        view = preds
        if show == "Correct":
            view = preds[preds["Result"] == "✅ Correct"]
        elif show == "Wrong":
            view = preds[preds["Result"] == "❌ Wrong"]
        elif show == "Missed buyers":
            view = preds[(preds["Actual"] == BUY) & (preds["Predicted"] == NO_BUY)]

        st.dataframe(
            view, width="stretch", height=420,
            column_config={
                "Purchase chance": st.column_config.ProgressColumn(
                    "Purchase chance", format="percent", min_value=0.0, max_value=1.0
                ),
                "PageValues": st.column_config.NumberColumn(format="%.2f"),
                "BounceRates": st.column_config.NumberColumn(format="%.3f"),
                "ExitRates": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        full = raw.copy()
        full["Predicted_Revenue"] = y_pred
        full["Purchase_Probability"] = res["y_prob"].round(4)
        st.caption(f"Showing {len(view):,} of {len(preds):,} sessions. "
                   "A purchase chance of 50 % or more counts as a predicted purchase.")
        st.download_button("Download predictions (CSV)", full.to_csv(index=False).encode(),
                           file_name=f"predictions_{MODELS[model_name][0].replace('.joblib', '')}.csv",
                           mime="text/csv", icon=":material/download:")


def compare_page():
    st.title("Compare all models")
    data_status()
    X, y, _, _ = DATA

    all_results = {name: evaluate(name, X, y) for name in MODELS}
    table = pd.DataFrame({n: r["metrics"] for n, r in all_results.items()}).T
    table.index.name = "ML Model Name"
    best = table["MCC"].idxmax()

    st.success(
        f"**{best}** performs best overall — MCC {table.loc[best, 'MCC']:.4f}, "
        f"F1 {table.loc[best, 'F1']:.4f}, accuracy {table.loc[best, 'Accuracy']:.1%}.",
        icon=":material/trophy:",
    )
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="rgba(124, 92, 191, 0.35)"),
        width="stretch",
    )
    st.caption("Highlighted = best model for that metric. MCC and F1 matter most here because only "
               "~16 % of sessions end in a purchase, so accuracy alone can look good for a weak model.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Metric by model")
        metric = st.selectbox("Metric", list(METRIC_HELP), index=5, label_visibility="collapsed")
        st.bar_chart(table[metric].sort_values(ascending=False), horizontal=True,
                     color=ACCENT, height=280)
        st.caption(METRIC_HELP[metric])
    with right:
        st.markdown("##### ROC curves")
        if y.nunique() == 2:
            st.pyplot(plot_roc(y, all_results))
        else:
            st.info("ROC curves need both purchase and no-purchase rows in the data.")


# ---------------------------------------------------------------- navigation + shared data
st.session_state.setdefault("model_choice", "Decision Tree")

page = st.navigation([
    st.Page(overview_page, title="Overview", icon=":material/home:", default=True),
    st.Page(evaluate_page, title="Evaluate a model", icon=":material/analytics:"),
    st.Page(compare_page, title="Compare all models", icon=":material/leaderboard:"),
])

with st.sidebar:
    st.markdown("**Test data**")
    data_choice = st.radio("Test data", ["Sample test data", "Upload CSV"],
                           label_visibility="collapsed", horizontal=True)
    uploaded = None
    if data_choice == "Upload CSV":
        uploaded = st.file_uploader("CSV with the 17 features + Revenue", type="csv")
    elif SAMPLE_CSV.exists():
        st.download_button("Download sample CSV", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv",
                           icon=":material/download:", width="stretch")
    st.caption("Models are pre-trained and loaded from disk. "
               "The test data is only used for evaluation — nothing is retrained.")

DATA = None
if data_choice == "Upload CSV":
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source = uploaded.name
    else:
        raw = None
else:
    raw, source = pd.read_csv(SAMPLE_CSV), "Sample test data"

if raw is not None:
    try:
        X_data, y_data = prepare(raw)
        DATA = (X_data, y_data, raw, source)
    except ValueError as err:
        st.sidebar.error(str(err))

page.run()
