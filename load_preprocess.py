import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# loading data
dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)
filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name = "filepaths").astype(str)
labels = pd.Series(labels, name = "label")

image_df = pd.concat([filepaths, labels], axis=1)

# data visualization
if __name__ == "__main__":
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    random_index = np.random.randint(0, len(image_df), 16)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))

    for i,  ax in enumerate(axes.flat):
        ax.imshow(plt.imread(image_df.filepaths[random_index[i]]))
        ax.set_title(image_df.label[random_index[i]])

    plt.tight_layout()
    grid_path = out_dir / "sample_grid_16.png"
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    class_counts = image_df["label"].value_counts().sort_index()

    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(1, 1, 1)

    ax2.bar(class_counts.index, class_counts.values)
    ax2.set_title("Class Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("#Images")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    dist_path = out_dir / "class_distribution.png"
    plt.savefig(dist_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig2)


# preprocessing
train_df, test_df = train_test_split(image_df, test_size = 0.2, shuffle = True, random_state=42)
 
# data augmentation + generators
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepaths",
    y_col="label",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="training"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepaths",
    y_col="label",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="validation"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepaths",
    y_col="label",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=False
)

def get_generators():
    return train_images, val_images, test_images