import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    import os
    import pandas as pd

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "master_df.csv")

    df = pd.read_csv(csv_path)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    return df

master_df = load_data()
# ==========================
# Executive KPIs
# ==========================

total_revenue = master_df["payment_value"].sum()
total_orders = master_df["order_id"].nunique()
total_customers = master_df["customer_id"].nunique()
average_order = master_df["payment_value"].mean()
ml_df = master_df.dropna(subset=["review_score", "payment_value"]).copy()
ml_df["good_review"] = (ml_df["review_score"] >= 4).astype(int)
X = ml_df[["payment_value"]]
y = ml_df["good_review"]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.set_page_config(
    page_title="E-commerce Executive Dashboard",
    layout="wide"
)

#BASE_DIR = os.path.dirname(os.path.dirname(__file__))
#csv_path = os.path.join(BASE_DIR, "data", "master_df.csv")

#master_df = pd.read_csv(csv_path)

st.title("📊 E-commerce Executive Dashboard")

st.sidebar.title("Dashboard Menu")

menu = st.sidebar.selectbox(
    "Choose Dashboard",
    [
        "Overview",
        "Sales Analysis",
        "Customer Analysis",
        "Seller Analysis",
        "Review Analysis",
        "Machine Learning"
    ]
)

states = sorted(master_df["customer_state"].dropna().unique())

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + list(states)
)

payment_types = sorted(master_df["payment_type"].dropna().unique())

selected_payment = st.sidebar.selectbox(
    "Payment Type",
    ["All"] + payment_types
)

filtered_df = master_df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["customer_state"] == selected_state
    ]

if selected_payment != "All":
    filtered_df = filtered_df[
        filtered_df["payment_type"] == selected_payment
    ]

min_date = filtered_df["order_purchase_timestamp"].min().date()
max_date = filtered_df["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

if len(date_range) == 2:
    start_date, end_date = date_range

    filtered_df = filtered_df[
        (filtered_df["order_purchase_timestamp"].dt.date >= start_date) &
        (filtered_df["order_purchase_timestamp"].dt.date <= end_date)
    ]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Revenue",
    f"R$ {filtered_df['payment_value'].sum():,.2f}"
)

col2.metric(
    "🛒 Orders",
    filtered_df['order_id'].nunique()
)

col3.metric(
    "👥 Customers",
    filtered_df['customer_unique_id'].nunique()
)

col4.metric(
    "📦 Products",
    filtered_df['product_id'].nunique()
)

#st.subheader("Revenue by State")

state_sales = (
    filtered_df
    .groupby("customer_state")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

if menu == "Overview":
    st.title("📊 Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
    "💰 Total Revenue",
    f"${total_revenue:,.2f}"
    )

    col2.metric(
    "🛒 Total Orders",
    f"{total_orders:,}"
    )

    col3.metric(
    "👥 Customers",
    f"{total_customers:,}"
    )

    col4.metric(
    "💳 Avg Order",
    f"${average_order:,.2f}"
    )

    st.markdown("---")
    
    st.header("Overview")

    st.subheader("Revenue by State")

    state_sales = (
        filtered_df.groupby("customer_state")["payment_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))
    state_sales.plot(kind="bar", ax=ax)
    st.pyplot(fig)


    st.subheader("Monthly Sales Trend")

    filtered_df["order_purchase_timestamp"] = pd.to_datetime(
        filtered_df["order_purchase_timestamp"]
    )

    filtered_df["Month"] = (
        filtered_df["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_sales = (
        filtered_df.groupby("Month")["payment_value"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(12,5))
    monthly_sales.plot(ax=ax)
    st.pyplot(fig)

elif menu == "Sales Analysis":

    st.header("Sales Analysis")

    payment_sales = (
        filtered_df.groupby("payment_type")["payment_value"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8,5))
    payment_sales.plot(kind="bar", ax=ax)

    st.pyplot(fig)

    st.subheader("Top Product Categories")

    category_sales = (
    filtered_df
    .groupby("product_category_name")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))
    category_sales.plot(kind="bar", ax=ax)

    st.pyplot(fig)


elif menu == "Customer Analysis":

    st.header("Customer Analysis")

    customers = (
        filtered_df["customer_state"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8,5))
    customers.plot(kind="bar", ax=ax)

    st.pyplot(fig)

elif menu == "Seller Analysis":

    st.header("Seller Analysis")

    seller_sales = (
    filtered_df
    .groupby("seller_id")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))
    seller_sales.plot(kind="bar", ax=ax)

    plt.xticks(rotation=90)

    st.pyplot(fig)

elif menu == "Review Analysis":

    st.header("Review Analysis")

    review_scores = (
    filtered_df["review_score"]
    .value_counts()
    .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8,5))
    review_scores.plot(kind="bar", ax=ax)

    st.pyplot(fig)

elif menu == "Machine Learning":

    st.title("Machine Learning Prediction")

    payment = st.number_input(
        "Enter Payment Value",
        min_value=0.0,
        value=100.0
    )

    prediction = model.predict([[payment]])

    if prediction[0] == 1:
        st.success("Customer is likely to give a GOOD review.")
    else:
        st.error("Customer is likely to give a BAD review.")

st.write(f"You selected: *{menu}*")

csv = filtered_df.to_csv(index=False)

st.download_button(
    "Download Filtered Data",
    csv,
    file_name="filtered_data.csv",
    mime="text/csv"
)

