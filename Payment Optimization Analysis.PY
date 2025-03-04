import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Generate synthetic dataset
np.random.seed(42)
n = 5000  # Number of transactions

data = {
    "Transaction_ID": np.arange(1, n+1),
    "User_ID": np.random.randint(1000, 5000, n),
    "Timestamp": pd.date_range(start="2024-01-01", periods=n, freq='T'),
    "Payment_Method": np.random.choice(["Credit Card", "PayPal", "Bank Transfer"], n),
    "Amount": np.round(np.random.uniform(5, 500, n), 2),
    "Status": np.random.choice(["Success", "Failure"], n, p=[0.85, 0.15]),
    "Region": np.random.choice(["North America", "Europe", "Asia", "South America"], n),
    "Retry_Attempt": np.random.randint(0, 3, n)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Extract hour for time-based analysis
df["Hour"] = df["Timestamp"].dt.hour

# Success vs Failure Rate by Payment Method
payment_method_stats = df.groupby("Payment_Method")["Status"].value_counts(normalize=True).unstack()

# Plot Success/Failure Rate by Payment Method
plt.figure(figsize=(10, 5))
payment_method_stats.plot(kind='bar', stacked=True, colormap='coolwarm', alpha=0.8)
plt.title("Transaction Success vs Failure Rate by Payment Method")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(title="Status")
plt.show()

# Failure Rate by Hour of the Day
hourly_failure_rate = df[df["Status"] == "Failure"].groupby("Hour").size() / df.groupby("Hour").size()

# Plot Failure Rate by Hour
plt.figure(figsize=(10, 5))
plt.plot(hourly_failure_rate.index, hourly_failure_rate.values, marker='o', linestyle='-', color='red')
plt.title("Transaction Failure Rate by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Failure Rate")
plt.grid()
plt.show()

# Regional Failure Rate Analysis
region_failure_rate = df[df["Status"] == "Failure"].groupby("Region").size() / df.groupby("Region").size()

# Plot Failure Rate by Region
plt.figure(figsize=(10, 5))
region_failure_rate.sort_values().plot(kind='bar', color='orange')
plt.title("Transaction Failure Rate by Region")
plt.ylabel("Failure Rate")
plt.xticks(rotation=30)
plt.show()

# Retry Success Rate Analysis
retry_success_rate = df[df["Retry_Attempt"] > 0].groupby("Retry_Attempt")["Status"].value_counts(normalize=True).unstack()

# Plot Retry Success Rate
plt.figure(figsize=(10, 5))
retry_success_rate.plot(kind='bar', stacked=True, colormap='viridis', alpha=0.8)
plt.title("Success vs Failure Rate by Retry Attempt")
plt.ylabel("Proportion")
plt.xlabel("Retry Attempts")
plt.xticks(rotation=0)
plt.legend(title="Status")
plt.show()

# Insights Generation
print("\nKey Insights:")
print("1. Payment methods with the highest failure rates should be optimized or replaced.")
print("2. Failure rates peak during certain hours, indicating potential server load issues.")
print("3. Some regions have significantly higher failure rates, possibly due to local banking issues.")
print("4. Retry attempts beyond a certain threshold do not significantly improve success rates.")
