# source /home/shay/a/dani/Nutrition_AI/.venv/bin/activate
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 #using predetermined model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout #layers in neural network
from tensorflow.keras.optimizers import Adam #backpropogation
from tensorflow.keras.callbacks import EarlyStopping
import os

#set constants
#IMAGES = "/home/shay/a/dani/Nutrition_AI/food-101/images"
IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-101/images" 
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16  # reduced from 32 to expedite training on CPU
TOTAL_CLASSES = 101
FREEZE_EPOCHS = 80
UNFREEZE_EPOCHS = 20
SEED = 123

#autotune data to optimize parallel processing
AUTOTUNE = tf.data.AUTOTUNE

#load data
train_data = tf.keras.utils.image_dataset_from_directory(
    IMAGES,
    validation_split=0.1, #because we are training on 100,000+ images, I will use 10% for validation and not 20%
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    IMAGES,
    validation_split=0.1,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

#because we automatically load data in, then use data augmentation to randomly flip, rotate, and zoom images for better training
data_augmentation = tf.keras.Sequential([ #groups layers into a model
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2)
])

#scales color of images to 0-1 and implements the flipping/rotation/zoom (normalization)
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = data_augmentation(image)
    return image, label

def preprocess_val(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_data = train_data.map(preprocess_train, num_parallel_calls=AUTOTUNE) #by using parallel calls, we can leverage multiple cpu cores and compute on separate cores
val_data = val_data.map(preprocess_val, num_parallel_calls=AUTOTUNE)

# Speed up pipeline using cache
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #caches images so faster to retrieve, shuffles a buffer to randomly sample, and prefetches data while the model is training in a previous batch
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

#using the MobileNetV2 model
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(TOTAL_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

#compile and train the model for the frozen dataset
model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training with frozen base model...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=FREEZE_EPOCHS,
    callbacks=[early_stop]
)

#compile and train the model for the unfrozen dataset
print("Unfreezing and fine-tuning model...")
base_model.trainable = True

model.compile(
    optimizer=Adam(1e-5), #note: lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=UNFREEZE_EPOCHS,
    callbacks=[early_stop]
)

#saving model
model_path = "mobilenetv2_food101_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")


# (.venv) bash-4.4$ /home/shay/a/dani/Nutrition_AI/.venv/bin/python /home/shay/a/dani/Nutrition_AI/mobilenetv2.py
# 2025-08-11 16:33:21.177570: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/thinlinc/lib64:/opt/thinlinc/lib
# 2025-08-11 16:33:21.177604: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
# Found 101000 files belonging to 101 classes.
# Using 90900 files for training.
# 2025-08-11 16:33:25.836932: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/thinlinc/lib64:/opt/thinlinc/lib
# 2025-08-11 16:33:25.836973: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
# 2025-08-11 16:33:25.836999: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ececomp4.ecn.purdue.edu): /proc/driver/nvidia/version does not exist
# 2025-08-11 16:33:25.837393: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 101000 files belonging to 101 classes.
# Using 10100 files for validation.
# Training with frozen base model...
# Epoch 1/80
# 2025-08-11 16:33:30.327027: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
# 5682/5682 [==============================] - 312s 54ms/step - loss: 3.9119 - accuracy: 0.1240 - val_loss: 2.8192 - val_accuracy: 0.3508
# Epoch 2/80
# 5682/5682 [==============================] - 306s 54ms/step - loss: 3.0862 - accuracy: 0.2631 - val_loss: 2.4190 - val_accuracy: 0.4175
# Epoch 3/80
# 5682/5682 [==============================] - 301s 53ms/step - loss: 2.8182 - accuracy: 0.3193 - val_loss: 2.2695 - val_accuracy: 0.4442
# Epoch 4/80
# 5682/5682 [==============================] - 286s 50ms/step - loss: 2.6587 - accuracy: 0.3522 - val_loss: 2.1825 - val_accuracy: 0.4563
# Epoch 5/80
# 5682/5682 [==============================] - 244s 43ms/step - loss: 2.5449 - accuracy: 0.3775 - val_loss: 2.1355 - val_accuracy: 0.4688
# Epoch 6/80
# 5682/5682 [==============================] - 240s 42ms/step - loss: 2.4552 - accuracy: 0.3951 - val_loss: 2.0797 - val_accuracy: 0.4779
# Epoch 7/80
# 5682/5682 [==============================] - 260s 46ms/step - loss: 2.3836 - accuracy: 0.4116 - val_loss: 2.0684 - val_accuracy: 0.4827
# Epoch 8/80
# 5682/5682 [==============================] - 250s 44ms/step - loss: 2.3287 - accuracy: 0.4235 - val_loss: 2.0340 - val_accuracy: 0.4877
# Epoch 9/80
# 5682/5682 [==============================] - 242s 43ms/step - loss: 2.2762 - accuracy: 0.4334 - val_loss: 2.0330 - val_accuracy: 0.4889
# Epoch 10/80
# 5682/5682 [==============================] - 244s 43ms/step - loss: 2.2294 - accuracy: 0.4423 - val_loss: 2.0026 - val_accuracy: 0.4977
# Epoch 11/80
# 5682/5682 [==============================] - 240s 42ms/step - loss: 2.1871 - accuracy: 0.4529 - val_loss: 2.0205 - val_accuracy: 0.4927
# Epoch 12/80
# 5682/5682 [==============================] - 238s 42ms/step - loss: 2.1483 - accuracy: 0.4620 - val_loss: 2.0044 - val_accuracy: 0.4968
# Epoch 13/80
# 5682/5682 [==============================] - 237s 42ms/step - loss: 2.1087 - accuracy: 0.4686 - val_loss: 1.9981 - val_accuracy: 0.4983
# Epoch 14/80
# 5682/5682 [==============================] - 242s 43ms/step - loss: 2.0746 - accuracy: 0.4749 - val_loss: 1.9938 - val_accuracy: 0.5015
# Epoch 15/80
# 5682/5682 [==============================] - 244s 43ms/step - loss: 2.0424 - accuracy: 0.4805 - val_loss: 1.9800 - val_accuracy: 0.5034
# Epoch 16/80
# 5682/5682 [==============================] - 237s 42ms/step - loss: 2.0074 - accuracy: 0.4903 - val_loss: 1.9791 - val_accuracy: 0.5027
# Epoch 17/80
# 5682/5682 [==============================] - 237s 42ms/step - loss: 1.9818 - accuracy: 0.4945 - val_loss: 2.0110 - val_accuracy: 0.4992
# Epoch 18/80
# 5682/5682 [==============================] - 236s 42ms/step - loss: 1.9564 - accuracy: 0.4997 - val_loss: 1.9859 - val_accuracy: 0.5054
# Epoch 19/80
# 5682/5682 [==============================] - 232s 41ms/step - loss: 1.9297 - accuracy: 0.5047 - val_loss: 2.0021 - val_accuracy: 0.5018
# Epoch 20/80
# 5682/5682 [==============================] - 228s 40ms/step - loss: 1.9015 - accuracy: 0.5102 - val_loss: 1.9787 - val_accuracy: 0.5105
# Epoch 21/80
# 5682/5682 [==============================] - 229s 40ms/step - loss: 1.8745 - accuracy: 0.5172 - val_loss: 1.9913 - val_accuracy: 0.5049
# Epoch 22/80
# 5682/5682 [==============================] - 237s 42ms/step - loss: 1.8495 - accuracy: 0.5197 - val_loss: 2.0196 - val_accuracy: 0.5031
# Epoch 23/80
# 5682/5682 [==============================] - 232s 41ms/step - loss: 1.8264 - accuracy: 0.5247 - val_loss: 2.0141 - val_accuracy: 0.4997
# Epoch 24/80
# 5682/5682 [==============================] - 238s 42ms/step - loss: 1.8028 - accuracy: 0.5316 - val_loss: 2.0029 - val_accuracy: 0.5065
# Epoch 25/80
# 5682/5682 [==============================] - 237s 42ms/step - loss: 1.7871 - accuracy: 0.5326 - val_loss: 2.0202 - val_accuracy: 0.5038
# Unfreezing and fine-tuning model...
# Epoch 1/20
# 5682/5682 [==============================] - 1042s 183ms/step - loss: 2.7439 - accuracy: 0.3532 - val_loss: 2.7082 - val_accuracy: 0.3533
# Epoch 2/20
# 5682/5682 [==============================] - 1120s 197ms/step - loss: 2.2279 - accuracy: 0.4534 - val_loss: 2.5534 - val_accuracy: 0.3940
# Epoch 3/20
# 5682/5682 [==============================] - 1117s 197ms/step - loss: 1.9969 - accuracy: 0.4993 - val_loss: 2.5490 - val_accuracy: 0.3993
# Epoch 4/20
# 5682/5682 [==============================] - 1116s 196ms/step - loss: 1.8241 - accuracy: 0.5348 - val_loss: 2.4268 - val_accuracy: 0.4231
# Epoch 5/20
# 5682/5682 [==============================] - 1106s 195ms/step - loss: 1.6846 - accuracy: 0.5649 - val_loss: 2.4359 - val_accuracy: 0.4250
# Epoch 6/20
# 5682/5682 [==============================] - 1115s 196ms/step - loss: 1.5567 - accuracy: 0.5891 - val_loss: 2.4530 - val_accuracy: 0.4322
# Epoch 7/20
# 5682/5682 [==============================] - 1102s 194ms/step - loss: 1.4518 - accuracy: 0.6134 - val_loss: 2.5110 - val_accuracy: 0.4256
# Epoch 8/20
# 5682/5682 [==============================] - 1093s 192ms/step - loss: 1.3490 - accuracy: 0.6360 - val_loss: 2.4665 - val_accuracy: 0.4386
# Epoch 9/20
# 5682/5682 [==============================] - 1091s 192ms/step - loss: 1.2572 - accuracy: 0.6573 - val_loss: 2.5570 - val_accuracy: 0.4296
# /home/shay/a/dani/Nutrition_AI/.venv/lib64/python3.6/site-packages/keras/utils/generic_utils.py:497: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
#   category=CustomMaskWarning)
# Model saved to mobilenetv2_food101_model.h5