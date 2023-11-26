# my_functions.py

import concurrent.futures
from functools import partial
from tqdm import tqdm
from scipy.spatial import distance
from scipy.sparse import csr_matrix
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils import add_self_loops

import subprocess
import psutil
import time
import logging

# Function to check memory usage
def check_memory_usage():
    virtual_memory = psutil.virtual_memory()
    used_memory_gb = virtual_memory.used / (1024**3)
    return used_memory_gb

# Function to construct a graph for a single frame
def construct_graph_for_frame_wrapper(frame_data, max_memory_gb):
    # Check memory usage before constructing the graph
    used_memory_gb = check_memory_usage()
    if used_memory_gb >= max_memory_gb:
        return None  # Return None if memory usage exceeds the limit

    try:
        threshold = compute_threshold(frame_data)  # Compute the threshold
        return construct_graph_for_frame(frame_data, threshold)
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return None  # Return None for failed frames

# Function to compute the threshold for a frame
def compute_threshold(frame_data):
    # Compute the threshold based on your logic here
    # For example, you can use the mean distance between centers
    centers = np.column_stack([(frame_data['bbox_left'] + frame_data['bbox_width'] / 2),
                               (frame_data['bbox_top'] + frame_data['bbox_height'] / 2)])
    centers_sparse = csr_matrix(centers)
    distances = distance.pdist(centers_sparse.toarray())
    threshold = np.mean(distances)
    return threshold
import time

def retry_construct_graph_for_frame(frame_data, max_retries=3):
    for _ in range(max_retries):
        try:
            threshold = compute_threshold(frame_data)  # Compute the threshold
            return construct_graph_for_frame(frame_data, threshold)
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
    return None  # Return None after max retries

max_memory_gb = 2
# Generate graphs frame by frame
unique_frames_train = df_train['frame_num'].unique()
unique_frames_test = df_test['frame_num'].unique()
unique_frames_val = df_val['frame_num'].unique()

for frame in unique_frames_train:
    train_graphs.append(construct_graph_for_frame(df_train[df_train['frame_num'] == frame]))

for frame in unique_frames_test:
    test_graphs.append(construct_graph_for_frame(df_test[df_test['frame_num'] == frame]))

for frame in unique_frames_val:
    val_graphs.append(construct_graph_for_frame(df_val[df_val['frame_num'] == frame]))
