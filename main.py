import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="SmartSpend - AI Expense Tracker", layout="wide")

# Initialize or load the expenses CSV file
def load_expenses():
    if os.path.exists('expenses.csv'):
        return pd.read_csv('expenses.csv')
    return pd.DataFrame(columns=['date', 'amount', 'category', 'payment_method', 'notes'])

# Categories and payment methods
CATEGORIES = ['Food', 'Rent', 'Transportation', 'Entertainment', 'Utilities', 'Shopping', 'Healthcare', 'Others']
PAYMENT_METHODS = ['Cash', 'Credit Card', 'Debit Card', 'Bank Transfer', 'Mobile Payment']

def add_expense(date, amount, category, payment_method, notes):
    df = load_expenses()
    new_expense = pd.DataFrame({
        'date': [date],
        'amount': [amount],
        'category': [category],
        'payment_method': [payment_method],
        'notes': [notes]
    })
    df = pd.concat([df, new_expense], ignore_index=True)
    df.to_csv('expenses.csv', index=False)
    return df

def analyze_spending_behavior(df):
    if len(df) < 3:  # Need at least 3 entries for meaningful clustering
        return "Not enough data for analysis"
    
    # Prepare data for clustering
    category_pivot = pd.pivot_table(df, values='amount', index=df.index, columns='category', fill_value=0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(category_pivot)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Determine spending behavior based on total amount
    total_spending = df['amount'].sum()
    avg_per_transaction = total_spending / len(df)
    
    if clusters[0] == 0:
        return "Frugal Spender"
    elif clusters[0] == 1:
        return "Balanced Spender"
    else:
        return "Heavy Spender"

def generate_saving_tips(df):
    tips = []
    total_spending = df['amount'].sum()
    
    # Analyze category-wise spending
    category_spending = df.groupby('category')['amount'].sum()
    for category, amount in category_spending.items():
        percentage = (amount / total_spending) * 100
        if percentage > 30:
            tips.append(f"You spent {percentage:.1f}% on {category}. Consider reducing expenses in this category.")
    
    return tips

def main():
    st.title("SmartSpend - AI Expense Tracker")
    st.markdown("*Track, Analyze, and Save Smarter*")
    
    # Sidebar for adding expenses
    st.sidebar.header("Add New Expense")
    date = st.sidebar.date_input("Date", datetime.today())
    amount = st.sidebar.number_input("Amount", min_value=0.0, value=0.0)
    category = st.sidebar.selectbox("Category", CATEGORIES)
    payment_method = st.sidebar.selectbox("Payment Method", PAYMENT_METHODS)
    notes = st.sidebar.text_area("Notes (Optional)")
    
    if st.sidebar.button("Add Expense"):
        df = add_expense(date, amount, category, payment_method, notes)
        st.sidebar.success("Expense added successfully!")
    
    # Main content
    df = load_expenses()
    
    if not df.empty:
        # Display recent expenses
        st.header("Recent Expenses")
        st.dataframe(df.tail())
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending by Category")
            fig, ax = plt.subplots()
            category_spending = df.groupby('category')['amount'].sum()
            ax.pie(category_spending, labels=category_spending.index, autopct='%1.1f%%')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Monthly Spending Trend")
            df['date'] = pd.to_datetime(df['date'])
            monthly_spending = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            fig, ax = plt.subplots()
            monthly_spending.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Spending behavior analysis
        st.header("Spending Analysis")
        behavior = analyze_spending_behavior(df)
        st.write(f"Your spending behavior: {behavior}")
        
        # Saving tips
        st.header("Smart Saving Tips")
        tips = generate_saving_tips(df)
        for tip in tips:
            st.info(tip)
    else:
        st.info("No expenses recorded yet. Add your first expense using the sidebar!")

if __name__ == "__main__":
    main()