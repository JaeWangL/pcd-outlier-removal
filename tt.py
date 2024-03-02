import pandas as pd

# Load your CSV file
df = pd.read_csv('results_old.csv')  # Replace 'your_file.csv' with your actual file path

# Calculate the new column
# df['diff_mine_target'] = ((df['mean_diff_mine'] - df['mean_diff_target']).abs() /
#                           ((df['mean_diff_mine'] + df['mean_diff_target']) / 2)) * 100
df = df.drop(['filename', 'better_target'], axis=1)
df['mean_diff_raw'] = df['mean_diff_raw'].round(2)
df['mean_diff_mine'] = df['mean_diff_mine'].round(2)
df['mean_diff_target'] = df['mean_diff_target'].round(2)

df.to_csv('results_old_p.csv')
