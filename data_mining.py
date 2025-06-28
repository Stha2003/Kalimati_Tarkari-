# ----------- data_mining.py -----------

from operator import le
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def prepare_data(df, commodity):
    df = df[df["Commodity"] == commodity].copy()
    df["Price_Category"] = pd.qcut(df["Average"], q=2, labels=["Low", "High"])
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Day"] = pd.to_datetime(df["Date"]).dt.day
    df["Weekday"] = pd.to_datetime(df["Date"]).dt.weekday

    features = ["Month", "Day", "Weekday"]
    df.dropna(subset=features + ["Price_Category"], inplace=True)

    X = df[features]
    y = df["Price_Category"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return train_test_split(X, y_encoded, test_size=0.3, random_state=42), le, features


def evaluate_model(y_true, y_pred, label_encoder):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    support = pd.Series(y_true).value_counts(sort=False)

    labels = label_encoder.classes_
    report_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "support": support
    }, index=labels)

    accuracy = (y_true == y_pred).mean()
    macro_avg = report_df[["precision", "recall", "f1-score"]].mean()
    weighted_avg = (report_df[["precision", "recall", "f1-score"]]
                    .multiply(report_df["support"], axis=0).sum() / report_df["support"].sum())

    report_df.loc["accuracy"] = ["", "", "", accuracy * len(y_true)]
    report_df.loc["macro avg"] = [*macro_avg, len(y_true)]
    report_df.loc["weighted avg"] = [*weighted_avg, len(y_true)]

    return report_df


def run_decision_tree(df, commodity):
    (X_train, X_test, y_train, y_test), le, features = prepare_data(df, commodity)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Decision Tree Report")

    report_df = evaluate_model(y_test, y_pred, le)
    st.dataframe(report_df.style.format(precision=2))

    # st.markdown("Graph :")
    # fig, ax = plt.subplots()
    # pd.Series(model.feature_importances_, index=features).sort_values().plot.barh(ax=ax)
    # ax.set_facecolor('#1e1e2f')
    # fig.patch.set_facecolor('#1e1e2f')
    # ax.tick_params(colors='white')
    # ax.title.set_color('white')
    # st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


def run_random_forest(df, commodity):
    (X_train, X_test, y_train, y_test), le, features = prepare_data(df, commodity)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Random Forest Report")

    report_df = evaluate_model(y_test, y_pred, le)
    st.dataframe(report_df.style.format(precision=2))

    # st.markdown("Graph :")
    # fig, ax = plt.subplots()
    # pd.Series(model.feature_importances_, index=features).sort_values().plot.barh(ax=ax)
    # ax.set_facecolor('#1e1e2f')
    # fig.patch.set_facecolor('#1e1e2f')
    # ax.tick_params(colors='white')
    # ax.title.set_color('white')
    # st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)


def run_random_forest_regression(df, selected_commodity):
    df = df[df["Commodity"] == selected_commodity].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    X = df[["Month", "Day", "Weekday"]]
    y = df["Average"]

    if len(df) < 10:
        st.warning(f"Not enough data to run regression for {selected_commodity}.")
        return

    # Keep the date index
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    date_test = df.loc[X_test.index, "Date"]

    # model = RandomForestRegressor(random_state=100, n_estimators= 200)
    model = AdaBoostRegressor(n_estimators= 100, learning_rate= 0.4, loss= 'square')
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)

    st.subheader("Average Price Prediction (Random Forest Regressor)")
    st.markdown(f"**RMSE**: {rmse:.2f} Rs")
    st.markdown(f"**R² Score**: {r2:.2f}")

    result_df = pd.DataFrame({
        "Date": date_test,
        "Actual": y_test.values,
        "Predicted": predicted
    }).sort_values("Date")

    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result_df["Date"], y=result_df["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=result_df["Date"], y=result_df["Predicted"], mode="lines+markers", name="Predicted"))
    fig.update_layout(
        title="Actual vs Predicted Average Price",
        xaxis_title="Date",
        yaxis_title="Price (Rs)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

