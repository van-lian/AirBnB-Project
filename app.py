import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("airbnb_cleaned.csv")
    return df

df = load_data()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")

neighbourhoods = st.sidebar.multiselect(
    "Select Neighbourhood(s):",
    options=df["neighbourhood"].unique(),
    default=df["neighbourhood"].unique()
)

room_types = st.sidebar.multiselect(
    "Select Room Type(s):",
    options=df["room_type"].unique(),
    default=df["room_type"].unique()
)

df_filtered = df[(df["neighbourhood"].isin(neighbourhoods)) &
                 (df["room_type"].isin(room_types))]

# ----------------------------
# Dashboard Title
# ----------------------------
st.title(" Airbnb Pricing Insights Dashboard")
st.markdown("Helping new hosts price competitively in NYC neighborhoods")

# ----------------------------
# Bar Chart: Avg Price by Neighbourhood & Room Type
# ----------------------------
avg_price = (
    df_filtered.groupby(["neighbourhood", "room_type"])["price"]
    .mean()
    .reset_index()
)

fig_bar = px.bar(
    avg_price,
    x="neighbourhood",
    y="price",
    color="room_type",
    barmode="group",
    title="Average Price by Neighbourhood and Room Type"
)
st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------
# Box Plot: Price Distribution by Room Type
# ----------------------------
fig_box = px.box(
    df_filtered,
    x="room_type",
    y="price",
    color="room_type",
    points="all",
    title="Price Distribution by Room Type"
)
st.plotly_chart(fig_box, use_container_width=True)

# ----------------------------
# Map: Price Heatmap (lat/long + price category)
# ----------------------------
fig_map = px.scatter_mapbox(
    df_filtered,
    lat="latitude",
    lon="longitude",
    color="price_category",
    size="price",
    mapbox_style="carto-positron",
    zoom=10,
    title="Airbnb Price Heatmap"
)
st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# Scatter: Price vs Property Age
# ----------------------------
fig_scatter = px.scatter(
    df_filtered,
    x="property_age",
    y="price",
    color="room_type",
    title="Price vs Property Age",
    trendline="ols"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------
# Extra: Key Metrics
# ----------------------------
st.subheader("📊Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Listings", f"{len(df_filtered):,}")
col2.metric("Avg Price", f"${df_filtered['price'].mean():.2f}")
col3.metric("Median Nights", f"{df_filtered['minimum_nights'].median():.0f}")
