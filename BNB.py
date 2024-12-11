# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
import warnings  
import os 
warnings.filterwarnings('ignore')

# %%
col_names = [
    'timestamp_start', 'open', 'high', 'low', 'close', 'volume', 
    'timestamp_end', 'quote_asset_volume', 'number_of_trades', 
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# %%
bnb_df = pd.read_csv('full_BNBUSDT_1m.csv')
bnb_df.columns = col_names


# Set 'timestamp_start' as the index
bnb_df.set_index('timestamp_start', inplace=True)
bnb_df.head()

# %%
bnb = pd.read_csv('full_BNBUSDT_1m.csv')

bnb.columns = col_names
bnb['timestamp_start'] = pd.to_datetime(bnb['timestamp_start'], unit='ms')
bnb.set_index('timestamp_start', inplace=True)
bnb['log_return'] = np.log(bnb['close'] / bnb['close'].shift(1))
bnb.head()

# %%
bnb.tail()

# %%
from datetime import datetime

# Define start and end times
start_time = datetime(2020, 1, 1, 0, 1, 0)
end_time = datetime(2024, 11, 30, 23, 59, 0)

# Calculate difference in days
delta = end_time - start_time
total_days = delta.days

# Convert total days to minutes
total_minutes = total_days * 1440 + (end_time.hour * 60 + end_time.minute) - (start_time.hour * 60 + start_time.minute)

total_days, total_minutes

# %%
pos_ts, neg_ts = pd.read_csv("positive_timestamps.csv"), pd.read_csv("negative_timestamps.csv")
pos_ts.columns = ['timestamp']
neg_ts.columns = ['timestamp']
pos_ts.head()  # now in df 

# %%
pos_ts_array = np.array(pos_ts['timestamp'])  
neg_ts_array = np.array(neg_ts['timestamp'])  
pos_ts_array

# %%
print(len(pos_ts_array) == len(neg_ts_array))
print(len(pos_ts_array))

# %%
bnb_df['log_return'] = np.log(bnb_df['close'] / bnb_df['close'].shift(1))

bnb_df.head()

# %%
""" change here to look at different window """ 


pos_ts_start  = pos_ts_array + (0 * 60 * 1000)
pos_ts_end = pos_ts_start + (5 * 60 * 1000)
neg_ts_start  = neg_ts_array + (0 * 60 * 1000)
neg_ts_end = neg_ts_start + (5 * 60 * 1000)

# %%
# type : a list of list 
pos_ts1_5 = [[start, end] for start, end in zip(pos_ts_start, pos_ts_end)]
neg_ts1_5 = [[start, end] for start, end in zip(neg_ts_start, neg_ts_end)]


# %%
pos_ts1_5_arr = np.array(pos_ts1_5)  # Shape: (n, 2)
neg_ts1_5_arr = np.array(neg_ts1_5)  # Shape: (n, 2)

# %%
# POS 
bnb_timestamps = bnb_df.index.values  # Get the timestamp_start values as an array

valid_ranges = []
for start, end in pos_ts1_5_arr:
    # Filter rows within the range
    filtered_range = bnb_df.loc[start:end, 'log_return']
    if not filtered_range.empty:
        valid_ranges.append((start, end, filtered_range.tolist()))

# Process results into a structured list
result1 = [{"range": [start, end], "log_returns": log_returns} for start, end, log_returns in valid_ranges]


for item in result1:
     print(f"Range {item['range']} -> Log Returns: {item['log_returns']}")


# %%
# NEG 

bnb_timestamps = bnb_df.index.values  # Get the timestamp_start values as an array

valid_ranges = []
for start, end in neg_ts1_5_arr:
    # Filter rows within the range
    filtered_range = bnb_df.loc[start:end, 'log_return']
    if not filtered_range.empty:
        valid_ranges.append((start, end, filtered_range.tolist()))

# Process results into a structured list
result2 = [{"range": [start, end], "log_returns": log_returns} for start, end, log_returns in valid_ranges]

for item in result2:
     print(f"Range {item['range']} -> Log Returns: {item['log_returns']}")

# %%
# Sum the first 5 log return
#POS
for item in result1:
    log_returns = item["log_returns"]
    # Sum the first 5 log returns
    ts1_5_log_ret = sum(log_returns[:-1])  
    item["ts1_5_sum"] = ts1_5_log_ret  # Add to the dictionary

for item in result1:
    print(f"Range {item['range']} -> Log Returns: {item['log_returns']}, Log Return sum: {item['ts1_5_sum']}")

# %%
result1

# %%
for item in result2:
    log_returns = item["log_returns"]
    ts1_5_log_ret = sum(log_returns)  
    item["ts1_5_sum"] = ts1_5_log_ret


# %%
ts1_5_log_ret = sum(log_returns[:5])
ts1_5_pos_sum = [item['ts1_5_sum'] for item in result1]
ts1_5_neg_sum = [item['ts1_5_sum'] for item in result2]
ts0_5_ts = [item['range'] for item in result1]
len(ts0_5_ts) 

# %% [markdown]
# ### Resample to 5min

# %%
bnb

# %%
bnb.index = pd.to_datetime(bnb.index)

# Resample the data to 5-minute intervals and sum all values
bnb_1min = bnb.resample('5T').sum()

# Print the resampled DataFrame
total_log_r = bnb_1min['log_return']
total_log_r

# %%
pos_jump = ts1_5_pos_sum
neg_jump = ts1_5_neg_sum

# %%
#POS
import seaborn as sns
# Plotting the histograms
plt.figure(figsize=(10, 6))

# Plot histogram for jump_log_r
sns.histplot(pos_jump, bins=1000, kde=False, color='blue', label='pos_jump_log_r', stat='density', edgecolor='black')

# Plot histogram for total_log_r
sns.histplot(total_log_r, bins=1000, kde=False, color='red', label='total_log_r', stat='density', edgecolor='black')

# Customize the plot
plt.axvline(0, color='gray', linestyle='--', label='Zero')
plt.title('BNB/USDT Histograms of log returns (Jump Log and Total Log)')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.legend()
plt.xlim(-0.03, 0.03)

# Show plot
plt.show()

# %%
# NEG

plt.figure(figsize=(10, 6))

# Plot histogram for jump_log_r
sns.histplot(neg_jump, bins=2000, kde=False, color='blue', label='neg_jump_log_r', stat='density', edgecolor='black')

# Plot histogram for total_log_r
sns.histplot(total_log_r, bins=2000, kde=False , color='red', label='total_log_r', stat='density', edgecolor='black')

# Customize the plot
plt.axvline(0, color='gray', linestyle='--', label='Zero')
plt.title('BNB/USDT Histograms of log returns (Jump Log and Total Log)')
plt.xlabel('Log Return')
plt.ylabel('Density')
plt.legend()
plt.xlim(-0.03, 0.03)

# Show plot
plt.show()

# %%
len(neg_jump), len(pos_jump), len(total_log_r)

# %% [markdown]
# ### Stats

# %%
from scipy import stats

def calculate_statistics(data):
    # Basic statistics
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Calculate percentiles
    percentiles = np.percentile(data, [1, 5, 25, 75, 95, 99])
    
    # Print results
    print("\nBasic Statistics:")
    print(f"Mean: {mean:.6f}")
    print(f"Median: {median:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Minimum: {min_val:.6f}")
    print(f"Maximum: {max_val:.6f}")
    
    print("\nShape Statistics:")
    print(f"Skewness: {skewness:.6f}")
    print(f"Kurtosis: {kurtosis:.6f}")
    
    print("\nPercentiles:")
    print(f"1st percentile: {percentiles[0]:.6f}")
    print(f"5th percentile: {percentiles[1]:.6f}")
    print(f"25th percentile: {percentiles[2]:.6f}")
    print(f"75th percentile: {percentiles[3]:.6f}")
    print(f"95th percentile: {percentiles[4]:.6f}")
    print(f"99th percentile: {percentiles[5]:.6f}")
    
    # Count positive and negative values
    positive_count = sum(x > 0 for x in data)
    negative_count = sum(x < 0 for x in data)
    zero_count = sum(x == 0 for x in data)
    
    print("\nValue Counts:")
    print(f"Positive values: {positive_count}")
    print(f"Negative values: {negative_count}")
    print(f"Zero values: {zero_count}")
    print(f"Total values: {len(data)}")


# Calculate statistics
calculate_statistics(neg_jump)

# %%
def analyze_distribution(data):
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Histogram with KDE
    sns.histplot(data, kde=True, ax=ax1)
    ax1.set_title('Histogram with Kernel Density Estimation')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Count')
    
    # 2. Q-Q Plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    # 3. Box Plot
    sns.boxplot(data, ax=ax3)
    ax3.set_title('Box Plot')
    
    # 4. Violin Plot
    sns.violinplot(data, ax=ax4)
    ax4.set_title('Violin Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests for normality
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ks_stat, ks_p = stats.kstest(data, 'norm')
    
    # Calculate distribution statistics
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    print("\nDistribution Analysis:")
    print("-" * 50)
    print(f"Mean: {mean:.6f}")
    print(f"Median: {median:.6f}")
    print(f"Standard Deviation: {std:.6f}")
    print(f"Skewness: {skewness:.6f}")
    print(f"Kurtosis: {kurtosis:.6f}")
    
    print("\nNormality Tests:")
    print("-" * 50)
    print(f"Shapiro-Wilk test:")
    print(f"Statistic: {shapiro_stat:.6f}")
    print(f"p-value: {shapiro_p:.6f}")
    print(f"\nKolmogorov-Smirnov test:")
    print(f"Statistic: {ks_stat:.6f}")
    print(f"p-value: {ks_p:.6f}")
    
    # Interpretation
    print("\nDistribution Interpretation:")
    print("-" * 50)
    print("Skewness interpretation:")
    if abs(skewness) < 0.5:
        print("The distribution is approximately symmetric")
    elif skewness > 0:
        print("The distribution is positively skewed (right-tailed)")
    else:
        print("The distribution is negatively skewed (left-tailed)")
        
    print("\nKurtosis interpretation:")
    if abs(kurtosis) < 0.5:
        print("The distribution has normal tail weight")
    elif kurtosis > 0:
        print("The distribution is leptokurtic (heavy-tailed)")
    else:
        print("The distribution is platykurtic (light-tailed)")
        
    print("\nNormality test interpretation:")
    alpha = 0.05
    if shapiro_p < alpha:
        print("The data significantly deviates from normal distribution")
    else:
        print("The data appears to follow a normal distribution")


analyze_distribution(pos_jump)

# %% [markdown]
# ### Intraday Analysis

# %%
ts0_5_ts = np.array(ts0_5_ts)
ts0_5_ts

# %%
# Convert the Unix timestamps to pandas datetime objects (start times)
bnb_ts_btcj_start = pd.to_datetime(ts0_5_ts[:, 0], unit='ms')
bnb_ts_btcj_end = pd.to_datetime(ts0_5_ts[:, 1], unit='ms')

# Create a time range from the very first to the very last timestamp with 1-minute intervals
time_range = pd.date_range(start=bnb_ts_btcj_start.min(), end=bnb_ts_btcj_end.max(), freq='T')

# Plotting the vertical lines
plt.figure(figsize=(8, 6))

# Plot vertical lines for each timestamp in sui_ts_btcj (converted from ts0_5_ts)
for ts in bnb_ts_btcj_start:
    plt.axvline(ts, color='blue', linewidth=0.1, alpha=0.8, linestyle='--')
for ts in bnb_ts_btcj_end:
    plt.axvline(ts, color='orange', linewidth=0.1, alpha=0.6, linestyle='--')
# Customize the plot
plt.title('Vertical Lines for pos_ts_start Timestamps')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.xlim(time_range[0], time_range[-1])
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()

# %%
bnb_ts_btcj_start

# %%
# Convert the Unix timestamps to pandas datetime objects (start times)
bnb_ts_btcj_start = pd.to_datetime(ts0_5_ts[:, 0], unit='ms')

# Set the date as x-axis (each tick is a day)
bnb_ts_btcj_start_dates = bnb_ts_btcj_start.date

# Create a DataFrame to count the frequency of timestamps for each day
date_counts = pd.Series(bnb_ts_btcj_start_dates).value_counts().sort_index()

# Generate a complete date range from the minimum to the maximum date
date_range = pd.date_range(start=bnb_ts_btcj_start.min().date(), end=bnb_ts_btcj_start.max().date(), freq='D')

# Reindex date_counts to include all days in the date_range
date_counts = date_counts.reindex(date_range.date, fill_value=0)

# Plotting the frequency of timestamps per day
plt.figure(figsize=(12, 6))
date_counts.plot(kind='bar', color='blue', alpha=0.7)

# Customize the plot
plt.title('Frequency of Timestamps Per Day')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')
plt.tight_layout()

# Show plot
plt.show()


