import pandas as pd
from multiprocessing import Pool
import os
from data_processing import preprocess_data  # Import the preprocess_data function from your script
import time

start_time = time.time()

# Determine the number of CPU cores available
num_cores = os.cpu_count()

# Define the file paths
train_file_path = r"C:\Users\1933358\OneDrive - Cesi\CoRODDA\SDD_train.csv"
test_file_path = r"C:\Users\1933358\OneDrive - Cesi\CoRODDA\SDD_test.csv"
val_file_path = r"C:\Users\1933358\OneDrive - Cesi\CoRODDA\SDD_val.csv"

# Use multiprocessing.Pool with the number of processes you want
num_processes = 24  # Adjust the number of processes as needed
with Pool(processes=num_processes) as pool:
    train_data = pool.map(preprocess_data, [train_file_path])
    test_data = pool.map(preprocess_data, [test_file_path])
    val_data = pool.map(preprocess_data, [val_file_path])

# Filter out None values (tasks that raised exceptions)
train_data = [data for data in train_data if data is not None]
test_data = [data for data in test_data if data is not None]
val_data = [data for data in val_data if data is not None]

# Concatenate the processed data
df_train = pd.concat(train_data, ignore_index=True)
df_test = pd.concat(test_data, ignore_index=True)
df_val = pd.concat(val_data, ignore_index=True)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")
