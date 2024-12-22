import pandas as pd
import numpy as np
import os
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Find .npy files in dataset directory. Each file contains a dataset for a given label, e.g. 'dataset/desert.npy'

def create_dataset():
    DATASET_BASE_DIR = '../datasets'
    DATASET_TARGET_DIR = './dataset/terrain/scaled/'

    datasets = {}
    for filename in os.listdir(DATASET_BASE_DIR):
        if filename.endswith('.npy'):
            label = filename.split('.')[0]
            datasets[label] = np.load(f'{DATASET_BASE_DIR}/{filename}')
            print(f"Loaded {label} dataset with shape {datasets[label].shape}")

    # Merge datasets into a single one
    X = np.vstack([datasets[label] for label in datasets])
    y = np.concatenate([[label] * datasets[label].shape[0] for label in datasets])
    label_names = list(datasets.keys())

    # Transform y to use hot-one encoding
    df_y = pd.get_dummies(y)
    df_y.head()
    y = df_y.to_numpy(dtype=np.float32)

    class_counts = [np.sum(y[:, i]) for i in range(y.shape[1])]
    print(f"Class counts: {class_counts}")

    print(X.shape, y.shape)


    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Flow into directories
    os.makedirs(DATASET_TARGET_DIR, exist_ok=True)

    # Save train
    train_dir = os.path.join(DATASET_TARGET_DIR, 'train')
    os.makedirs(train_dir, exist_ok=True)
    for label in label_names:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)

    print("Saving training images...")

    for i, (x, y) in enumerate(zip(X_train, y_train)):
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
        x = x.astype(np.uint8)
        image = Image.fromarray(x)
        image = image.convert('L')
        y_label = np.argmax(y)
        y_label_name = label_names[y_label]
        image.save(os.path.join(train_dir, f'{y_label_name}{i}.png'))

    print(f"Saved {len(X_train)} training images")

    # Save val
    val_dir = os.path.join(DATASET_TARGET_DIR, 'val')
    os.makedirs(val_dir, exist_ok=True)
    for label in label_names:
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)

    print("Saving validation images...")

    for i, (x, y) in enumerate(zip(X_val, y_val)):
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
        x = x.astype(np.uint8)
        image = Image.fromarray(x)
        image = image.convert('L')
        y_label = np.argmax(y)
        y_label_name = label_names[y_label]
        image.save(os.path.join(val_dir, f'{y_label_name}/{i}.png'))

    print(f"Saved {len(X_val)} validation images")

    # Save test
    test_dir = os.path.join(DATASET_TARGET_DIR, 'test')
    os.makedirs(test_dir, exist_ok=True)
    for label in label_names:
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    print("Saving test images...")

    for i, (x, y) in enumerate(zip(X_test, y_test)):
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
        x = x.astype(np.uint8)
        image = Image.fromarray(x)
        image = image.convert('L')
        y_label = np.argmax(y)
        y_label_name = label_names[y_label]
        image.save(os.path.join(test_dir, f'{y_label_name}/{i}.png'))

    print(f"Saved {len(X_test)} test images")


if __name__ == '__main__':
    create_dataset()