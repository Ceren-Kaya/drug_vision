import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from load_preprocess import get_generators

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

train_images, val_images, test_images = get_generators()
num_classes = len(train_images.class_indices)

# build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights = "imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape =(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss = "categorical_crossentropy",
    metrics=["accuracy"]
)

# callbacks
checkp_path = OUTPUT_DIR / "best_model.keras"
callbacks = [ 
    ModelCheckpoint(checkp_path, save_weights_only=False, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience = 5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
]

# training 
history1 = model.fit(
    train_images,
    validation_data=val_images,
    epochs = 10,
    callbacks=callbacks
)

# finetuning
base_model.trainable = True

for layer in base_model.layers[:120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_images,
    validation_data=val_images,
    epochs = 20,
    initial_epoch=10,
    callbacks=callbacks
)

# visualize training curves
def plot_curves(history, out_path):
    hist = history.history
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(hist["accuracy"], label="train_acc")
    ax1.plot(hist["val_accuracy"], label="val_acc")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(hist["loss"], label="train_loss")
    ax2.plot(hist["val_loss"], label="val_loss")
    ax2.set_title("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

plot_curves(history2, FIG_DIR / "training_curves.png")