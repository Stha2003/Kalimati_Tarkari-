# data_mining.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def prepare_data(df, commodity):
    df = df[df["Commodity"] == commodity].copy()

    df["Price_Category"] = pd.qcut(df["Average"], q=2, labels=["Low", "High"])
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Day"] = pd.to_datetime(df["Date"]).dt.day
    df["Weekday"] = pd.to_datetime(df["Date"]).dt.weekday
    features = ["Month", "Day", "Weekday", "Maximum", "Minimum"]
    df.dropna(subset=features + ["Price_Category"], inplace=True)

    X = df[features]
    y = df["Price_Category"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return train_test_split(X, y_encoded, test_size=0.3, random_state=42), le, features


def run_decision_tree(df, commodity):
    (X_train, X_test, y_train, y_test), le, features = prepare_data(df, commodity)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(" Decision Tree Report")

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    st.markdown(f"<div class='model-output'>{report}</div>", unsafe_allow_html=True)

    st.markdown("Graph :")
    fig, ax = plt.subplots()
    pd.Series(model.feature_importances_, index=features).sort_values().plot.barh(ax=ax)
    ax.set_facecolor('#1e1e2f')
    fig.patch.set_facecolor('#1e1e2f')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


def run_random_forest(df, commodity):
    (X_train, X_test, y_train, y_test), le, features = prepare_data(df, commodity)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(" Random Forest Report")

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    st.markdown(f"<div class='model-output'>{report}</div>", unsafe_allow_html=True)

    st.markdown("Graph :")
    fig, ax = plt.subplots()
    pd.Series(model.feature_importances_, index=features).sort_values().plot.barh(ax=ax)
    ax.set_facecolor('#1e1e2f')
    fig.patch.set_facecolor('#1e1e2f')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
