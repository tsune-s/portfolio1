"""Main Streamlit dashboard application."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import DataLoader
from src.visualization.dashboard_utils import DashboardComponents

# Page config
st.set_page_config(
    page_title="Data Analysis Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


def load_data():
    """Load all datasets."""
    loader = DataLoader()
    try:
        churn_df = loader.load_csv("customer_churn.csv")
        sales_df = loader.load_csv("sales_data.csv")
        basket_df = loader.load_csv("market_basket.csv")
        return churn_df, sales_df, basket_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please run: `uv run python scripts/generate_sample_data.py`")
        return None, None, None


def show_overview():
    """Display overview page."""
    st.title("📊 Data Analysis Portfolio")
    st.markdown("---")

    st.markdown("""
    ## Welcome to My Data Analytics Portfolio

    This interactive dashboard showcases my expertise in **data analysis**, **machine learning**,
    and **data visualization**. Explore different projects using the sidebar navigation.

    ### 🎯 Key Skills Demonstrated

    #### Data Engineering
    - ETL pipeline design and implementation
    - Data quality validation and cleansing
    - Feature engineering and transformation

    #### Machine Learning
    - Supervised learning (classification, regression)
    - Model evaluation and optimization
    - Hyperparameter tuning
    - Feature importance analysis

    #### Data Visualization
    - Interactive dashboards with Streamlit
    - Statistical visualizations
    - Business intelligence reporting

    ### 📁 Available Projects

    1. **Customer Churn Prediction**
       - Advanced classification model
       - Feature importance analysis
       - Interactive predictions

    2. **Sales Forecasting**
       - Time series analysis
       - Trend and seasonality decomposition
       - Revenue predictions

    3. **Market Basket Analysis**
       - Association rule mining
       - Product recommendation engine
       - Network visualization

    ### 🛠️ Technology Stack

    - **Languages**: Python 3.12+
    - **Data Processing**: Pandas, Polars, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost, LightGBM
    - **Visualization**: Plotly, Matplotlib, Seaborn, Streamlit
    - **Database**: SQLAlchemy

    ---
    *Select a project from the sidebar to begin exploration.*
    """)


def show_churn_analysis(df):
    """Display customer churn analysis."""
    st.title("🔄 Customer Churn Prediction")
    st.markdown("Advanced predictive modeling to identify customers at risk of churning.")
    st.markdown("---")

    components = DashboardComponents()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")

    with col2:
        churn_rate = df['churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        avg_tenure = df['tenure_months'].mean()
        st.metric("Avg Tenure (months)", f"{avg_tenure:.0f}")

    with col4:
        avg_charges = df['monthly_charges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

    st.markdown("---")

    # Two columns layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Churn by Contract Type")
        churn_by_contract = df.groupby(['contract_type', 'churn']).size().reset_index(name='count')
        fig = px.bar(
            churn_by_contract,
            x='contract_type',
            y='count',
            color='churn',
            title='Customer Distribution by Contract Type',
            labels={'churn': 'Churned', 'count': 'Number of Customers'},
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Monthly Charges Distribution")
        fig = px.histogram(
            df,
            x='monthly_charges',
            color='churn',
            title='Distribution of Monthly Charges',
            labels={'churn': 'Churned'},
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📞 Support Calls vs Churn")
        fig = px.box(
            df,
            x='churn',
            y='num_support_calls',
            title='Number of Support Calls by Churn Status',
            labels={'churn': 'Churned', 'num_support_calls': 'Support Calls'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⏱️ Tenure Analysis")
        fig = px.histogram(
            df,
            x='tenure_months',
            color='churn',
            title='Distribution of Customer Tenure',
            labels={'churn': 'Churned', 'tenure_months': 'Tenure (months)'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model performance section
    st.markdown("---")
    st.subheader("🤖 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "75.4%")
        st.metric("Precision", "71.6%")

    with col2:
        st.metric("Recall", "75.4%")
        st.metric("F1 Score", "72.3%")

    with col3:
        st.metric("ROC AUC", "75.5%")

    st.info("📝 The model was trained using XGBoost with 200 estimators and achieved strong performance on the test set.")


def show_sales_analysis(df):
    """Display sales analysis."""
    st.title("📈 Sales Forecasting")
    st.markdown("Time series analysis and forecasting for retail sales data.")
    st.markdown("---")

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_sales = df['total_sales'].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")

    with col2:
        avg_daily_sales = df['total_sales'].mean()
        st.metric("Avg Daily Sales", f"${avg_daily_sales:.0f}")

    with col3:
        max_sales = df['total_sales'].max()
        st.metric("Peak Daily Sales", f"${max_sales:.0f}")

    with col4:
        total_transactions = df['num_transactions'].sum()
        st.metric("Total Transactions", f"{total_transactions:,}")

    st.markdown("---")

    # Time series plot
    st.subheader("📊 Sales Trend Over Time")
    fig = px.line(
        df,
        x='date',
        y='total_sales',
        title='Daily Sales Revenue',
        labels={'total_sales': 'Sales ($)', 'date': 'Date'}
    )
    fig.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    # Category breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Sales by Category")
        category_cols = ['electronics_sales', 'clothing_sales', 'food_sales', 'home_sales']
        category_total = df[category_cols].sum()
        fig = px.pie(
            values=category_total.values,
            names=['Electronics', 'Clothing', 'Food', 'Home'],
            title='Revenue Distribution by Category'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Monthly Sales Trend")
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly_sales = df.groupby('month')['total_sales'].sum().reset_index()
        fig = px.bar(
            monthly_sales,
            x='month',
            y='total_sales',
            title='Monthly Revenue',
            labels={'total_sales': 'Sales ($)', 'month': 'Month'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_basket_analysis(df):
    """Display market basket analysis."""
    st.title("🛒 Market Basket Analysis")
    st.markdown("Association rule mining to discover purchasing patterns.")
    st.markdown("---")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        n_transactions = df['transaction_id'].nunique()
        st.metric("Total Transactions", f"{n_transactions:,}")

    with col2:
        n_products = df['product'].nunique()
        st.metric("Unique Products", n_products)

    with col3:
        avg_basket_size = df.groupby('transaction_id').size().mean()
        st.metric("Avg Basket Size", f"{avg_basket_size:.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔝 Top Selling Products")
        product_sales = df.groupby('product')['quantity'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=product_sales.values,
            y=product_sales.index,
            orientation='h',
            title='Top 10 Products by Quantity Sold',
            labels={'x': 'Quantity', 'y': 'Product'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔢 Items per Transaction")
        items_per_transaction = df.groupby('transaction_id').size()
        fig = px.histogram(
            x=items_per_transaction,
            title='Distribution of Basket Sizes',
            labels={'x': 'Number of Items', 'y': 'Frequency'},
            nbins=15
        )
        st.plotly_chart(fig, use_container_width=True)

    # Product frequency
    st.subheader("📊 Product Purchase Frequency")
    product_freq = df.groupby('product').size().sort_values(ascending=False)
    fig = px.bar(
        x=product_freq.index,
        y=product_freq.values,
        title='Number of Transactions per Product',
        labels={'x': 'Product', 'y': 'Transaction Count'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application."""
    # Sidebar
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Customer Churn", "Sales Forecasting", "Market Basket"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 About
    This portfolio demonstrates advanced data analysis and machine learning capabilities.

    ### 🔗 Links
    - [GitHub](https://github.com)
    - [LinkedIn](https://linkedin.com)

    ### 📧 Contact
    your.email@example.com
    """)

    # Load data
    if page != "Overview":
        churn_df, sales_df, basket_df = load_data()

        if churn_df is None:
            st.stop()

    # Display selected page
    if page == "Overview":
        show_overview()
    elif page == "Customer Churn":
        show_churn_analysis(churn_df)
    elif page == "Sales Forecasting":
        show_sales_analysis(sales_df)
    elif page == "Market Basket":
        show_basket_analysis(basket_df)


if __name__ == "__main__":
    main()
