# ----------- kalimati_dashboard/app.py -----------

import streamlit as st
import pandas as pd
import plotly.express as px
from Components.map import show_kalimati_map
from data_mining import run_decision_tree, run_random_forest, run_random_forest_regression

# ---------- 1. Page Config ----------
st.set_page_config(
    page_title="Kalimati Tarkari Dashboard",
    page_icon="🥕",
    layout="wide",
)

# ---------- 1.1 Load CSS ----------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True) 

# ---------- 2. Load & Prepare ----------
@st.cache_data
def load_data():
    df = pd.read_csv('Data/kalimati_tarkari_dataset_cleaned.csv')
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()
    return df

df = load_data()

# ---------- 3. Sidebar Filters ----------
st.sidebar.header("Filter Options")

commodities = sorted(df["Commodity"].unique())
years = sorted(df["Year"].unique(), reverse=True)

selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)
selected_year = st.sidebar.selectbox("Select Year", years)

price_label_to_col = {
    "Average Price": "Average",
    "Maximum Price": "Maximum",
    "Minimum Price": "Minimum",
}
selected_price_label = st.sidebar.radio("Price Type", list(price_label_to_col.keys()))
price_col = price_label_to_col[selected_price_label]

# ---------- 4. Filter Data ----------
filtered_df = df[
    (df["Commodity"] == selected_commodity) &
    (df["Year"] == selected_year)
].sort_values("Date")

# ---------- 5. Charts ----------
line_chart = px.line(
    filtered_df,
    x="Date",
    y=price_col,
    markers=True,
    labels={price_col: f"{selected_price_label} (Rs)", "Date": "Date"},
    title=f"{selected_price_label} Trend of {selected_commodity} in {selected_year}",
)
line_chart.update_layout(template="plotly_dark")

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
monthly_avg = (
    filtered_df.groupby("Month")[price_col]
    .mean()
    .reindex(month_order)
    .fillna(0)
)

bar_chart = px.bar(
    x=monthly_avg.index,
    y=monthly_avg.values,
    labels={"x": "Month", "y": f"Avg {selected_price_label} (Rs)"},
    color=monthly_avg.values,
    color_continuous_scale="viridis",
)
bar_chart.update_layout(template="plotly_dark")

# ---------- 6. Dashboard Layout ----------
st.title(f" Price Dashboard: {selected_commodity} ({selected_year})")
st.markdown("Source: Kalimati Tarkari Bazar")

col1, col2 = st.columns([1, 1])
with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"Daily {selected_price_label}")
        st.plotly_chart(line_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(" Monthly Average Price")
        st.plotly_chart(bar_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- 7. Modeling Section ----------
st.markdown("---")
st.header(" Price Category Prediction ")

model_col1, model_col2 = st.columns(2)
with model_col1:
    run_decision_tree(df, selected_commodity)

with model_col2:
    run_random_forest(df, selected_commodity)

# ---------- 7.5 Regression ----------
st.markdown("---")
st.subheader(" Predict Future Average Prices ")
run_random_forest_regression(df, selected_commodity)

# ---------- 8. Map and Raw Data ----------
bottom1, bottom2 = st.columns([1, 1])
with bottom1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        show_kalimati_map(df)
        st.markdown('</div>', unsafe_allow_html=True)

with bottom2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(" View Raw Data")
        st.dataframe(filtered_df)
        st.markdown('</div>', unsafe_allow_html=True)
