"""Generate synthetic datasets for portfolio projects."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_customer_churn_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer churn dataset.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with customer churn data
    """
    np.random.seed(42)

    # Customer demographics
    customer_ids = [f"CUST_{i:05d}" for i in range(n_samples)]
    age = np.random.normal(45, 15, n_samples).astype(int).clip(18, 80)
    gender = np.random.choice(['Male', 'Female'], n_samples)

    # Account information
    tenure_months = np.random.exponential(24, n_samples).astype(int).clip(1, 72)
    contract_type = np.random.choice(
        ['Month-to-Month', 'One Year', 'Two Year'],
        n_samples,
        p=[0.5, 0.3, 0.2]
    )

    # Services
    internet_service = np.random.choice(
        ['DSL', 'Fiber Optic', 'No'],
        n_samples,
        p=[0.4, 0.4, 0.2]
    )
    online_security = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    tech_support = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    streaming_tv = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])

    # Billing
    monthly_charges = np.random.normal(65, 30, n_samples).clip(20, 150)
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 50, n_samples)

    # Customer behavior
    num_support_calls = np.random.poisson(2, n_samples)
    payment_method = np.random.choice(
        ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'],
        n_samples,
        p=[0.4, 0.15, 0.25, 0.2]
    )

    # Generate churn based on features (realistic pattern)
    churn_prob = (
        0.05  # Base rate
        + 0.3 * (contract_type == 'Month-to-Month')
        + 0.15 * (num_support_calls > 3)
        + 0.2 * (tenure_months < 6)
        + 0.1 * (payment_method == 'Electronic Check')
        - 0.15 * (online_security == 'Yes')
        - 0.1 * (tech_support == 'Yes')
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.random(n_samples) < churn_prob).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'gender': gender,
        'tenure_months': tenure_months,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'online_security': online_security,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'num_support_calls': num_support_calls,
        'payment_method': payment_method,
        'churn': churn
    })

    return df


def generate_sales_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic sales time series dataset.

    Args:
        n_samples: Number of days to generate

    Returns:
        DataFrame with sales data
    """
    np.random.seed(42)

    # Generate dates
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]

    # Generate trend, seasonality, and noise
    trend = np.linspace(100, 200, n_samples)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    noise = np.random.normal(0, 10, n_samples)

    sales = trend + seasonal + weekly + noise
    sales = sales.clip(50, None)

    # Product categories
    product_categories = ['Electronics', 'Clothing', 'Food', 'Home']
    category_sales = {
        cat: (sales * np.random.uniform(0.7, 1.3, n_samples)).round(2)
        for cat in product_categories
    }

    df = pd.DataFrame({
        'date': dates,
        'total_sales': sales.round(2),
        **{f'{cat.lower()}_sales': vals for cat, vals in category_sales.items()},
        'num_transactions': np.random.poisson(50, n_samples),
        'avg_transaction_value': (sales / np.random.poisson(50, n_samples)).round(2)
    })

    return df


def generate_market_basket_data(n_transactions: int = 10000) -> pd.DataFrame:
    """Generate synthetic market basket dataset.

    Args:
        n_transactions: Number of transactions to generate

    Returns:
        DataFrame with transaction data
    """
    np.random.seed(42)

    products = [
        'Milk', 'Bread', 'Eggs', 'Butter', 'Cheese',
        'Yogurt', 'Coffee', 'Tea', 'Sugar', 'Flour',
        'Chicken', 'Beef', 'Pasta', 'Rice', 'Tomatoes',
        'Onions', 'Apples', 'Bananas', 'Orange Juice'
    ]

    # Association rules (products that are bought together)
    associations = {
        'Milk': ['Eggs', 'Bread', 'Butter'],
        'Coffee': ['Sugar', 'Milk'],
        'Pasta': ['Tomatoes', 'Onions'],
        'Bread': ['Butter', 'Eggs']
    }

    transactions = []
    for trans_id in range(n_transactions):
        # Random number of items
        n_items = np.random.randint(1, 8)

        # Start with random products
        basket = set(np.random.choice(products, n_items, replace=False))

        # Add associated products with some probability
        for product in list(basket):
            if product in associations:
                for assoc_product in associations[product]:
                    if np.random.random() < 0.6:
                        basket.add(assoc_product)

        # Create transaction rows
        for product in basket:
            transactions.append({
                'transaction_id': f'T{trans_id:06d}',
                'product': product,
                'quantity': np.random.randint(1, 5)
            })

    df = pd.DataFrame(transactions)
    return df


def main():
    """Generate all sample datasets."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample datasets...")

    # Customer Churn
    print("\n1. Generating customer churn dataset...")
    churn_df = generate_customer_churn_data(n_samples=5000)
    churn_path = data_dir / "customer_churn.csv"
    churn_df.to_csv(churn_path, index=False)
    print(f"   Saved to {churn_path}")
    print(f"   Shape: {churn_df.shape}")
    print(f"   Churn rate: {churn_df['churn'].mean():.2%}")

    # Sales Data
    print("\n2. Generating sales time series dataset...")
    sales_df = generate_sales_data(n_samples=1095)  # 3 years
    sales_path = data_dir / "sales_data.csv"
    sales_df.to_csv(sales_path, index=False)
    print(f"   Saved to {sales_path}")
    print(f"   Shape: {sales_df.shape}")
    print(f"   Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")

    # Market Basket
    print("\n3. Generating market basket dataset...")
    basket_df = generate_market_basket_data(n_transactions=10000)
    basket_path = data_dir / "market_basket.csv"
    basket_df.to_csv(basket_path, index=False)
    print(f"   Saved to {basket_path}")
    print(f"   Shape: {basket_df.shape}")
    print(f"   Unique transactions: {basket_df['transaction_id'].nunique()}")

    print("\n✅ All datasets generated successfully!")


if __name__ == "__main__":
    main()
