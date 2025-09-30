import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so we can import the `src` package when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
from src.inference.utils import load_model, predict_bookings, get_seasonal_insights

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Airbnb NYC Insights",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Better Styling
# ----------------------------
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5ea;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .filter-header {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    
    .chart-container {
        background-color: rgba(0, 0, 0, 0);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    h1 {
        color: #2e3d49;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stSelectbox > div > div {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Preprocessed Data
# ----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/airbnb_cleaned.csv")
        df.columns = df.columns.str.strip()
        
        # Calculate property age (current year - construction year)
        current_year = 2025  # Update this as needed
        df['property_age'] = current_year - df['Construction year']
        
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'airbnb_cleaned.csv' not found. Please upload the file.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ----------------------------
# Header Section
# ----------------------------
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🏠 Airbnb NYC Analytics Dashboard</h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: 0;">
            Smart pricing insights and demand forecasting for NYC Airbnb hosts
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Tab Navigation
# ----------------------------
tab1, tab2 = st.tabs(["📊 Pricing Analytics", "🔮 Demand Prediction"])

with tab1:
    # ----------------------------
    # Sidebar Filters 
    # ----------------------------
    with st.sidebar:
        st.markdown('<div class="filter-header">🔍 FILTER OPTIONS</div>', unsafe_allow_html=True)
        
        st.markdown("### 📍 Location")
        neighbourhoods = st.multiselect(
            "Select Neighbourhoods:",
            options=sorted(df["neighbourhood"].unique()),
            default=sorted(df["neighbourhood"].unique())[:3],
            help="Choose specific neighbourhoods to analyze"
        )
        
        st.markdown("### 🏡 Property Type")
        room_types = st.multiselect(
            "Select Room Types:",
            options=df["room type"].unique(),
            default=df["room type"].unique(),
            help="Filter by accommodation type"
        )
        
        st.markdown("### 💰 Price Range")
        if not df.empty:
            price_range = st.slider(
                "Price Range ($):",
                min_value=int(df["price"].min()),
                max_value=int(df["price"].quantile(0.95)),
                value=(int(df["price"].min()), int(df["price"].quantile(0.95))),
                step=10,
                help="Adjust price range (excludes extreme outliers)"
            )
        
        # Filter data
        if neighbourhoods and room_types:
            df_filtered = df[
                (df["neighbourhood"].isin(neighbourhoods)) &
                (df["room type"].isin(room_types)) &
                (df["price"] >= price_range[0]) &
                (df["price"] <= price_range[1])
            ]
        else:
            df_filtered = df[
                (df["price"] >= price_range[0]) &
                (df["price"] <= price_range[1])
            ]
        
        st.markdown("---")
        st.markdown(f"**📊 Showing {len(df_filtered):,} listings**")

    # ----------------------------
    # Key Metrics Cards
    # ----------------------------
    if not df_filtered.empty:
        st.markdown("## 📈 Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df_filtered['price'].mean()
            st.markdown(f"""
                <div class="metric-card">
                    <h3>💵 Average Price</h3>
                    <h2>${avg_price:.0f}</h2>
                    <p>per night</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            median_price = df_filtered['price'].median()
            st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Median Price</h3>
                    <h2>${median_price:.0f}</h2>
                    <p>per night</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_listings = len(df_filtered)
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🏠 Total Listings</h3>
                    <h2>{total_listings:,}</h2>
                    <p>properties</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_nights = df_filtered['minimum nights'].median()
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🗓️ Avg Min Nights</h3>
                    <h2>{avg_nights:.0f}</h2>
                    <p>nights minimum</p>
                </div>
            """, unsafe_allow_html=True)

        # ----------------------------
        # Charts Section
        # ----------------------------
        
        # Price by Neighbourhood (Horizontal Bar for better readability)
        st.markdown("## 🏙️ Price Analysis by Location")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            avg_price_neighbourhood = df_filtered.groupby("neighbourhood")["price"].agg(['mean', 'count']).reset_index()
            avg_price_neighbourhood = avg_price_neighbourhood.sort_values('mean', ascending=True)
            
            fig_neighbourhood = px.bar(
                avg_price_neighbourhood,
                x='mean',
                y='neighbourhood',
                orientation='h',
                title="Average Price by Neighbourhood",
                labels={'mean': 'Average Price ($)', 'neighbourhood': 'Neighbourhood'},
                color='mean',
                color_continuous_scale='Viridis',
                text='mean'
            )
            
            fig_neighbourhood.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
            fig_neighbourhood.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title=dict(x=0.5, font=dict(size=16)),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                yaxis=dict(showgrid=False)
            )
            
            st.plotly_chart(fig_neighbourhood, use_container_width=True)
        
        with col2:
            # Price distribution summary
            st.markdown("### 📊 Price Summary")
            price_stats = df_filtered['price'].describe()
            
            st.markdown(f"""
            **Price Statistics:**
            - **Min:** ${price_stats['min']:.0f}
            - **25%:** ${price_stats['25%']:.0f}
            - **50%:** ${price_stats['50%']:.0f}
            - **75%:** ${price_stats['75%']:.0f}
            - **Max:** ${price_stats['max']:.0f}
            - **Std:** ${price_stats['std']:.0f}
            """)
            
            # Room type distribution
            room_type_counts = df_filtered['room type'].value_counts()
            fig_donut = px.pie(
                values=room_type_counts.values,
                names=room_type_counts.index,
                hole=0.4,
                title="Room Type Distribution"
            )
            fig_donut.update_layout(height=250, showlegend=True, font=dict(size=10))
            st.plotly_chart(fig_donut, use_container_width=True)
        
        # Price Distribution Analysis
        st.markdown("## 💰 Price Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Box Plot
            fig_box = px.box(
                df_filtered,
                x="room type",
                y="price",
                color="room type",
                title="Price Distribution by Room Type",
                points="outliers"
            )
            
            fig_box.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(x=0.5, font=dict(size=16)),
                showlegend=False,
                xaxis=dict(title="Room Type", showgrid=False),
                yaxis=dict(title="Price ($)", showgrid=True, gridwidth=1, gridcolor='LightGray')
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Violin Plot for better distribution visualization
            fig_violin = px.violin(
                df_filtered,
                x="room type",
                y="price",
                color="room type",
                title="Price Density Distribution",
                box=True
            )
            
            fig_violin.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(x=0.5, font=dict(size=16)),
                showlegend=False,
                xaxis=dict(title="Room Type", showgrid=False),
                yaxis=dict(title="Price ($)", showgrid=True, gridwidth=1, gridcolor='LightGray')
            )
            
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # ----------------------------
        # Map Visualization (if coordinates available)
        # ----------------------------
        if "lat" in df_filtered.columns and "long" in df_filtered.columns:
            st.markdown("## 🗺️ Interactive Geographic Price Distribution")
            
            # Map controls
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                map_style = st.selectbox(
                    "Map Style:",
                    options=["carto-positron", "open-street-map", "carto-darkmatter", "stamen-terrain"],
                    index=0,
                    help="Choose your preferred map style"
                )
            
            with col2:
                zoom_level = st.slider(
                    "Initial Zoom:",
                    min_value=9,
                    max_value=13,
                    value=10,
                    help="Set the initial zoom level"
                )
            
            with col3:
                st.markdown("**💡 Map Tips:** Use mouse wheel to zoom, click and drag to pan, hover over dots for details")
            
            # Create price categories for better visualization
            df_map = df_filtered.copy()
            
            # Remove extreme outliers for better size scaling
            price_99th = df_map['price'].quantile(0.99)
            df_map = df_map[df_map['price'] <= price_99th]
            
            # Create price categories
            df_map['price_category'] = pd.cut(
                df_map['price'], 
                bins=5, 
                labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury']
            )
            
            # Add jitter to reduce overlapping
            np.random.seed(42)
            df_map['lat_jittered'] = df_map['lat'] + np.random.normal(0, 0.001, size=len(df_map))
            df_map['long_jittered'] = df_map['long'] + np.random.normal(0, 0.001, size=len(df_map))
            
            # Normalize price for size
            df_map['size_normalized'] = np.sqrt(df_map['price']) / 3
            
            # Sample data for performance if needed
            sample_size = min(2000, len(df_map))
            if len(df_map) > sample_size:
                df_map_display = df_map.sample(n=sample_size, random_state=42)
                st.info(f"📊 Showing {sample_size:,} out of {len(df_map):,} listings for optimal performance")
            else:
                df_map_display = df_map
            
            # Create map
            fig_map = px.scatter_map(
                df_map_display,
                lat="lat_jittered",
                lon="long_jittered",
                color="price_category",
                size="size_normalized",
                hover_name="neighbourhood",
                hover_data={
                    "room type": True,
                    "price": ":$,.0f",
                    "minimum nights": True,
                    "lat_jittered": False,
                    "long_jittered": False,
                    "size_normalized": False,
                    "price_category": False
                },
                map_style=map_style,
                zoom=zoom_level,
                title="Interactive Airbnb Price Map - Zoom & Explore!",
                color_discrete_map={
                    'Budget': '#2E8B57',
                    'Economy': '#32CD32',
                    'Standard': '#FFD700',
                    'Premium': '#FF8C00',
                    'Luxury': '#DC143C'
                },
                category_orders={
                    "price_category": ['Budget', 'Economy', 'Standard', 'Premium', 'Luxury']
                },
                size_max=15,
                opacity=0.7
            )
            
            fig_map.update_layout(
                height=600,
                title=dict(x=0, font=dict(size=18)),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255, 255, 255, 0)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1
                )
            )

            st.plotly_chart(fig_map, use_container_width=True)
            
            # Map statistics
            st.markdown("### 📊 Map Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>📍 Mapped Listings</h3>
                        <h2>{len(df_map_display):,}</h2>
                        <p>properties shown</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                avg_lat = df_map_display['lat'].mean()
                avg_lon = df_map_display['long'].mean()
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Map Center</h3>
                        <h2>{avg_lat:.3f}</h2>
                        <p>Lat: {avg_lat:.3f}, Lon: {avg_lon:.3f}</p>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                price_range_map = df_map_display['price'].max() - df_map_display['price'].min()
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>💰 Price Range</h3>
                        <h2>${price_range_map:,.0f}</h2>
                        <p>price spread</p>
                    </div>
                """, unsafe_allow_html=True)

        # Property Age Analysis
        st.markdown("## 🗓️ Property Age vs Price Analysis")
        
        # Check if property_age column exists and has data
        has_property_age = "property_age" in df_filtered.columns and df_filtered["property_age"].notna().sum() > 0
        
        if has_property_age:
            if not neighbourhoods:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem;
                        border-radius: 10px;
                        color: white;
                        text-align: center;
                        margin: 1rem 0;
                    ">
                        <h3>🏙️ Select Neighborhoods to View Analysis</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">
                            Please choose one or more neighborhoods from the sidebar filters to see how property age affects pricing in your selected areas.
                        </p>
                        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.9;">
                            💡 <strong>Tip:</strong> Try selecting specific neighborhoods like "Manhattan", "Brooklyn", or "Queens" to see detailed insights!
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                property_age_data = df_filtered.dropna(subset=['property_age'])
                
                if not property_age_data.empty and len(property_age_data) > 5:
                    fig_scatter = px.scatter(
                        property_age_data,
                        x="property_age",
                        y="price",
                        color="room type",
                        size="minimum nights",
                        title=f"Property Age vs Price Analysis - {', '.join(neighbourhoods[:3])}{'...' if len(neighbourhoods) > 3 else ''}",
                        trendline="ols",
                        hover_data=["neighbourhood"],
                        labels={
                            "property_age": "Property Age (years)",
                            "price": "Price per Night ($)",
                            "minimum nights": "Min Nights"
                        }
                    )
                    
                    fig_scatter.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title=dict(x=0.5, font=dict(size=16)),
                        xaxis=dict(title="Property Age (years)", showgrid=True, gridwidth=1, gridcolor='LightGray'),
                        yaxis=dict(title="Price per Night ($)", showgrid=True, gridwidth=1, gridcolor='LightGray')
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Add insights
                    st.markdown("### 📊 Key Insights")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        avg_age = property_age_data['property_age'].mean()
                        correlation = property_age_data['property_age'].corr(property_age_data['price'])
                        
                        st.markdown(f"""
                        **Age Analysis:**
                        - **Average Property Age:** {avg_age:.1f} years
                        - **Age-Price Correlation:** {correlation:.3f}
                        - **Correlation Strength:** {'Strong' if abs(correlation) > 0.5 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}
                        """)
                    
                    with col2:
                        newest_data = property_age_data[property_age_data['property_age'] <= 5]
                        oldest_data = property_age_data[property_age_data['property_age'] >= 20]
                        
                        newest_avg_price = newest_data['price'].mean() if len(newest_data) > 0 else 0
                        oldest_avg_price = oldest_data['price'].mean() if len(oldest_data) > 0 else 0
                        
                        st.markdown(f"""
                        **Price Trends:**
                        - **New Properties (≤5 years):** ${newest_avg_price:.0f} avg
                        - **Older Properties (≥20 years):** ${oldest_avg_price:.0f} avg
                        - **Age Premium:** {'Newer properties cost more' if newest_avg_price > oldest_avg_price else 'Older properties cost more' if oldest_avg_price > newest_avg_price else 'Similar pricing'}
                        """)
                else:
                    st.warning("⚠️ Not enough property age data available for selected neighborhoods.")
        else:
            st.info("📊 Property age data not available in the current dataset.")
        
        # Advanced Analytics Section
        st.markdown("## 🎯 Pricing Recommendations")
        
        if neighbourhoods:
            selected_neighbourhood = neighbourhoods[0]
            neighbourhood_data = df_filtered[df_filtered["neighbourhood"] == selected_neighbourhood]
            
            if not neighbourhood_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### 📍 {selected_neighbourhood} Insights")
                    
                    room_type_prices = neighbourhood_data.groupby("room type")["price"].agg(['mean', 'median', 'count']).round(2)
                    
                    st.markdown("**Recommended Pricing by Room Type:**")
                    for room_type, stats in room_type_prices.iterrows():
                        st.markdown(f"""
                        **{room_type}:**
                        - Average: ${stats['mean']:.0f}
                        - Median: ${stats['median']:.0f} 
                        - Listings: {stats['count']}
                        """)
                
                with col2:
                    st.markdown("### 💡 Pricing Strategy Tips")
                    
                    avg_price = neighbourhood_data['price'].mean()
                    competition = len(neighbourhood_data)
                    
                    st.markdown(f"""
                    **Market Analysis:**
                    - **Average Price:** ${avg_price:.0f}
                    - **Competition:** {competition} listings
                    - **Market Position:** {'High' if avg_price > df['price'].quantile(0.75) else 'Medium' if avg_price > df['price'].quantile(0.25) else 'Budget'} pricing area
                    
                    **Recommendations:**
                    - Price 10-15% below average for quick bookings
                    - Price at average for steady occupancy  
                    - Price 10-20% above for premium positioning
                    """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>📊 Dashboard built with Streamlit & Plotly | Data insights for smarter hosting decisions</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No data available with current filters. Please adjust your selection.")

# ----------------------------
# Prediction Tab
# ----------------------------
with tab2:
    st.markdown("## 🔮 Demand Prediction")
    st.markdown("Predict bookings and pricing for specific areas and seasons.")
    
    # Load the trained model
    try:
        with st.spinner("🔄 Loading trained model..."):
            model_data = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Prediction interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Select Area")
        area_options = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        selected_area = st.selectbox(
            "Choose NYC Borough:",
            options=area_options,
            help="Select the area for prediction"
        )
    
    with col2:
        st.markdown("### 🌿 Select Season")
        season_options = ['Winter', 'Spring', 'Summer', 'Fall']
        selected_season = st.selectbox(
            "Choose Season:",
            options=season_options,
            help="Select the season for prediction"
        )
    
    # Prediction button
    if st.button("🔮 Get Prediction", type="primary"):
        with st.spinner("🔄 Generating prediction..."):
            try:
                # Get prediction
                prediction = predict_bookings(selected_area, selected_season, model_data)
                
                # Display results
                st.markdown("## 📈 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Predicted Bookings</h3>
                            <h2>{prediction['predicted_bookings']}</h2>
                            <p>expected bookings</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>💰 Predicted Price</h3>
                            <h2>${prediction['predicted_price']}</h2>
                            <p>per night</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>🏠 Occupancy Rate</h3>
                            <h2>{prediction['occupancy_rate']}%</h2>
                            <p>expected occupancy</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Detailed insights
                st.markdown("### 📋 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Prediction Factors")
                    st.markdown(f"""
                    **Area Multiplier:** {prediction['area_multiplier']:.2f}x
                    - Manhattan: 1.2x (Premium area)
                    - Brooklyn: 1.0x (Standard)
                    - Queens: 0.8x (Budget-friendly)
                    - Bronx: 0.6x (Emerging)
                    - Staten Island: 0.7x (Suburban)
                    
                    **Seasonal Multiplier:** {prediction['seasonal_multiplier']:.2f}x
                    - Summer: 1.23x (Peak season)
                    - Spring: 0.90x (Moderate)
                    - Fall: 0.87x (Stable)
                    - Winter: 0.65x (Low season)
                    """)
                
                with col2:
                    st.markdown("#### 💡 Business Insights")
                    insights = get_seasonal_insights(selected_season)
                    if insights:
                        st.markdown(f"""
                        **Season Description:** {insights['description']}
                        
                        **Recommendation:** {insights['recommendation']}
                        
                        **Pricing Strategy:** {insights['pricing_strategy']}
                        
                        **Marketing Focus:** {insights['marketing_focus']}
                        """)
                
                # Visual representation
                st.markdown("### 📈 Visual Summary")
                
                fig = go.Figure()
                
                categories = ['Bookings', 'Price ($)', 'Occupancy (%)']
                values = [
                    prediction['predicted_bookings'],
                    prediction['predicted_price'],
                    prediction['occupancy_rate']
                ]
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    text=values,
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f"Prediction Summary: {selected_area} - {selected_season}",
                    xaxis_title="Metrics",
                    yaxis_title="Values",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional recommendations
                st.markdown("### 🚀 Action Items")
                
                if prediction['predicted_bookings'] > 120:
                    st.success("🎉 **High Demand Expected!** Consider increasing prices and preparing for peak occupancy.")
                elif prediction['predicted_bookings'] > 90:
                    st.info("📈 **Moderate Demand** - Good time for steady bookings with competitive pricing.")
                else:
                    st.warning("📉 **Low Demand Period** - Focus on attracting longer stays and business travelers.")
                
                # Price optimization tip
                if prediction['predicted_price'] > 700:
                    st.info("💡 **Premium Pricing Opportunity** - Your area commands higher prices. Consider luxury positioning.")
                elif prediction['predicted_price'] > 500:
                    st.info("💡 **Standard Pricing** - Competitive market positioning recommended.")
                else:
                    st.info("💡 **Budget-Friendly Market** - Focus on value proposition and volume.")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    # Footer for prediction tab
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>🔮 AI-powered demand forecasting | Optimize your Airbnb strategy with data-driven insights</p>
    </div>
    """, unsafe_allow_html=True)