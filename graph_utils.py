
from functools import partial
from tqdm import tqdm
from scipy.spatial import distance
from scipy.sparse import csr_matrix
import numpy as np
import torch
from torch_geometric.data import Data


# Set the GPU Device in PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a function to construct a graph for a single frame
def construct_graph_for_frame(frame_data, threshold):
    nodes = frame_data[['bbox_top', 'bbox_left', 'bbox_width', 'bbox_height']].values
    centers = np.column_stack([(frame_data['bbox_left'] + frame_data['bbox_width'] / 2),
                               (frame_data['bbox_top'] + frame_data['bbox_height'] / 2)])

    # Convert your centers to a sparse matrix
    centers_sparse = csr_matrix(centers)

    # Compute pairwise distances
    distances = distance.pdist(centers_sparse.toarray())

    spatial_edges = np.column_stack(np.where(distances < threshold))

    edge_index = torch.tensor(spatial_edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(nodes, dtype=torch.float).to(device)  # Move data to GPU if available
    y = torch.tensor(frame_data['object_pose'].values, dtype=torch.long).to(device)  # Move data to GPU if available
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Function to parallelize graph construction for multiple frames
def parallel_construct_graphs(unique_frames, df_data, num_processes, max_memory_gb):
    # Calculate the maximum number of frames to process in parallel based on available memory
    frame_size_gb = df_data.memory_usage(deep=True).sum() / (1024**3)
    max_frames = int(max_memory_gb / frame_size_gb)

    # Split frames into batches to avoid memory issues
    frame_batches = [unique_frames[i:i + max_frames] for i in range(0, len(unique_frames), max_frames)]

    # Create an empty list to store the resulting graphs
    graphs = []

    for batch in frame_batches:
        # Compute the threshold outside the parallel section
        centers_all = []
        for frame in batch:
            frame_data = df_data[df_data['frame_num'] == frame]
            centers = np.column_stack([(frame_data['bbox_left'] + frame_data['bbox_width'] / 2),
                                        (frame_data['bbox_top'] + frame_data['bbox_height'] / 2)])
            centers_all.append(centers)

        centers_all = np.vstack(centers_all)
        distances_all = distance.pdist(centers_all)
        threshold = np.mean(distances_all)

        # Use multiprocessing to construct graphs in parallel for this batch
        pool = multiprocessing.Pool(processes=num_processes)
        construct_graph_partial = partial(construct_graph_for_frame, threshold=threshold)
        batch_graphs = list(tqdm(pool.imap(construct_graph_partial, [df_data[df_data['frame_num'] == frame] for frame in batch]), total=len(batch)))

        # Extend the list of graphs with the graphs from this batch
        graphs.extend(batch_graphs)

    return graphs