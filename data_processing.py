# data_processing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)

        # Preprocessing steps
        df = df.iloc[1:, :]
        df.columns = ['frame_num', 'identity_num', 'bbox_top', 'bbox_left', 'bbox_width', 'bbox_height', 'object_category', 'occlusion', 'truncation', 'object_pose']

        # Define and fit the label encoder for 'object_pose' column
        le = LabelEncoder()
        df['object_pose'] = le.fit_transform(df['object_pose'])

        df[['bbox_top', 'bbox_left', 'bbox_width', 'bbox_height']] = df[['bbox_top', 'bbox_left', 'bbox_width', 'bbox_height']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['bbox_top', 'bbox_left', 'bbox_width', 'bbox_height'])

        return df

    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing {file_path}: {str(e)}")
        return None
