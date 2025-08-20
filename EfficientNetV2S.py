import tensorflow as tf
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

# ================================================================
# CONSTANTS
# ================================================================
BASE_DIR = "/Users/raunaksmac/Desktop/Nutrition AI/food-101"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
META_DIR = os.path.join(BASE_DIR, "meta")
IMAGE_SIZE = (224, 224)  # EfficientNetV2S uses 224x224
BATCH_SIZE = 32
TOTAL_CLASSES = 101
FREEZE_EPOCHS = 8
UNFREEZE_EPOCHS = 12
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE

# ================================================================
# LOAD OFFICIAL FOOD-101 SPLITS
# ================================================================
def load_official_splits():
    with open(os.path.join(META_DIR, "classes.txt"), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    with open(os.path.join(META_DIR, "train.json"), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(META_DIR, "test.json"), 'r') as f:
        test_data = json.load(f)
    return classes, train_data, test_data

classes, train_split, test_split = load_official_splits()
class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

print(f"Loaded {len(classes)} classes")
print(f"Train samples: {sum(len(images) for images in train_split.values())}")
print(f"Test samples: {sum(len(images) for images in test_split.values())}")

# ================================================================
# FILE PATH CREATION
# ================================================================
def create_file_paths_and_labels(data_split, split_name="train"):
    file_paths, labels, missing_files = [], [], 0
    print(f"\nProcessing {split_name} split...")

    for class_name, image_ids in data_split.items():
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(IMAGES_DIR, class_name)

        for image_id in image_ids:
            actual_image_id = image_id.split("/")[-1]
            file_path = os.path.join(class_dir, f"{actual_image_id}.jpg")
            if os.path.exists(file_path):
                file_paths.append(file_path)
                labels.append(class_idx)
            else:
                missing_files += 1
    print(f"{split_name} set: {len(file_paths)} images found, {missing_files} missing")
    return file_paths, labels

train_paths, train_labels = create_file_paths_and_labels(train_split, "Train")
test_paths, test_labels = create_file_paths_and_labels(test_split, "Test")

# Create validation split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels,
    test_size=0.2,
    random_state=SEED,
    stratify=train_labels
)

print(f"Final split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# ================================================================
# DATASET CREATION
# ================================================================
def create_tf_dataset(file_paths, labels, batch_size, training=False):
    labels_categorical = tf.keras.utils.to_categorical(labels, num_classes=TOTAL_CLASSES)
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_categorical))

    if training:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=SEED)

    def load_and_resize(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    dataset = dataset.map(load_and_resize, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

train_data = create_tf_dataset(train_paths, train_labels, BATCH_SIZE, training=True)
val_data = create_tf_dataset(val_paths, val_labels, BATCH_SIZE, training=False)
test_data = create_tf_dataset(test_paths, test_labels, BATCH_SIZE, training=False)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.02),
], name="data_augmentation")

# ================================================================
# PREPROCESSING
# ================================================================
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    image = data_augmentation(image, training=True)
    image = preprocess_input(image)  # EfficientNetV2S expects [0,1]
    return image, label

def preprocess_val(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

train_data = train_data.map(preprocess_train, num_parallel_calls=AUTOTUNE)
val_data = val_data.map(preprocess_val, num_parallel_calls=AUTOTUNE)
test_data = test_data.map(preprocess_val, num_parallel_calls=AUTOTUNE)

# Verify preprocessing
for images, _ in train_data.take(1):
    print(f"Preprocessed batch shape: {images.shape}")
    print(f"Range: {tf.reduce_min(images):.4f} – {tf.reduce_max(images):.4f}")
    print(f"Mean: {tf.reduce_mean(images):.4f}, Std: {tf.math.reduce_std(images):.4f}")

# ================================================================
# MODEL CREATION
# ================================================================
def create_model_efficientnet():
    inputs = Input(shape=(224, 224, 3), name="input_layer")
    base_model = EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None
    )
    base_model.trainable = False
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.2, seed=SEED)(x)
    outputs = Dense(TOTAL_CLASSES, activation="softmax", dtype="float32")(x)
    model = Model(inputs, outputs, name="food101_efficientnetv2s")
    return model, base_model

model, base_model = create_model_efficientnet()
model.summary()

# ================================================================
# TRAINING PHASE 1 (Frozen backbone)
# ================================================================
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", "top_k_categorical_accuracy"]
)

early_stop = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-8, mode="min")

print("\n=== Training Phase 1 (Frozen backbone) ===")
history_frozen = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FREEZE_EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ================================================================
# TRAINING PHASE 2 (Unfreeze fine-tuning)
# ================================================================
print("\n=== Training Phase 2 (Fine-tuning) ===")
base_model.trainable = True
total_layers = len(base_model.layers)
for i, layer in enumerate(base_model.layers):
    if i < int(total_layers * 0.8):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", "top_k_categorical_accuracy"]
)

early_stop_finetune = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max")
reduce_lr_finetune = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-9, mode="min")

history_unfreeze = model.fit(
    train_data,
    validation_data=val_data,
    epochs=UNFREEZE_EPOCHS,
    callbacks=[early_stop_finetune, reduce_lr_finetune],
    verbose=1
)

# ================================================================
# FINAL EVALUATION
# ================================================================
print("\n=== Final Evaluation on Test Set ===")
test_results = model.evaluate(test_data, verbose=1)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Top-5 Accuracy: {test_results[2]:.4f}")

# Save model and class mapping
model.save("food101_efficientnetv2s.h5")
print("Model saved to food101_efficientnetv2s.h5")

class_mapping = {
    "classes": classes,
    "class_to_idx": class_to_idx,
    "idx_to_class": {idx: cls for cls, idx in class_to_idx.items()}
}
with open("food101_class_mapping.pkl", "wb") as f:
    pickle.dump(class_mapping, f)
print("Class mapping saved to food101_class_mapping.pkl")
