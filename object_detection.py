import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
import random
from joblib import dump
import joblib
import copy
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import DataLoader, random_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset

import cv2



# Threshold for distance
threshold_distance = 10
in_channels = 3
hidden_channels = 16
num_classes_fl = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_classes = 2
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the pre-trained YOLO model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolo.eval()

# Load the trained GCN model
model_gsage = GraphSAGE(in_channels, hidden_channels, num_classes).to(device)
model_gsage.load_state_dict(torch.load('model_GraphSAGE.pth'))
model_gsage.eval()

# Load the FL model
model_fl = GraphSAGE(in_channels, hidden_channels, num_classes_fl).to(device)
model_fl.load_state_dict(torch.load('global_best_model.pth'))
model_fl.eval()

# Set the threshold for distance
threshold_distance = 10



        
       


# Function to perform object detection using YOLO only
def perform_yolo_object_detection(image, model_yolo):
    # Perform object detection using YOLOv5
    results_yolo = model_yolo(image)

    # Extract bounding box coordinates, class labels, and object confidence from YOLO results
    boxes_yolo = results_yolo.xyxy[0][:, :4].detach().cpu().numpy()
    labels_yolo = results_yolo.xyxy[0][:, 5].detach().cpu().numpy()
    confidences_yolo = results_yolo.xyxy[0][:, 4].detach().cpu().numpy()
    
        


    # Draw a rectangle and display the coordinates and accuracy for each detected object on the image
    for box_yolo, label_yolo, confidence_yolo in zip(boxes_yolo, labels_yolo, confidences_yolo):
        xmin, ymin, xmax, ymax = box_yolo.astype(int)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        st.write(f"Object: Label - {label_yolo}, Confidence - {confidence_yolo}")
        st.write(f"Coordinates: (xmin, ymin, xmax, ymax) - ({xmin}, {ymin}, {xmax}, {ymax})")

    # Display the image with rectangles
    st.image(image, channels="BGR")

# Create a Streamlit app
st.title("Object Detection with FedGNN_DAOD")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Perform object detection with YOLO if an image is uploaded
if uploaded_file is not None:
    # Load the image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform object detection using YOLO only
    perform_yolo_object_detection(image, model_yolo)