# kalimati_dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- 1. Page Config ----------
st.set_page_config(
    page_title="Kalimati Tarkari Dashboard",
    page_icon="🥕",
    layout="wide",
)

# ---------- 2. Load & Prepare ----------
@st.cache_data
def load_data():
    df = pd.read_csv('Data/kalimati_tarkari_dataset_cleaned.csv')
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name()
    return df

df = load_data()

# ---------- 3. Sidebar ----------
st.sidebar.header("Filter Options")

commodities = sorted(df["Commodity"].unique())
years       = sorted(df["Year"].unique(), reverse=True)

selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)
selected_year      = st.sidebar.selectbox("Select Year", years)

# Radio labels (nice for the UI) → actual column names
price_label_to_col = {
    "Average Price":  "Average",
    "Maximum Price":  "Maximum",
    "Minimum Price":  "Minimum",
}

selected_price_label = st.sidebar.radio("Price Type", list(price_label_to_col.keys()))
price_col            = price_label_to_col[selected_price_label]

# ---------- 4. Filter ----------
filtered_df = df[(df["Commodity"] == selected_commodity) & (df["Year"] == selected_year)].sort_values("Date")

# ---------- 5. Title ----------
st.title(f"📊 Price Dashboard: {selected_commodity} ({selected_year})")
st.markdown("Source: Kalimati Tarkari Bazar")

# ---------- 6. Daily Line Chart ----------
st.subheader(f"📈 Daily {selected_price_label}")
line_chart = px.line(
    filtered_df,
    x="Date",
    y=price_col,                          # <-- actual column name
    markers=True,
    labels={price_col: f"{selected_price_label} (Rs)", "Date": "Date"},
    title=f"{selected_price_label} Trend of {selected_commodity} in {selected_year}",
)
line_chart.update_layout(template="plotly_dark")
st.plotly_chart(line_chart, use_container_width=True)

# ---------- 7. Monthly Average Bar Chart ----------
st.subheader("📊 Monthly Average Price")

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
st.plotly_chart(bar_chart, use_container_width=True)

# ---------- 8. Static Map (optional) ----------
st.subheader("🗺 Kalimati Market Location")
kalimati_location = pd.DataFrame({"lat": [27.6973], "lon": [85.3065]})
st.map(kalimati_location)

# ---------- 9. Raw Data ----------
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df)
