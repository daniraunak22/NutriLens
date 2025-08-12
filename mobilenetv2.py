# source /home/shay/a/dani/Nutrition_AI/.venv/bin/activate
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 #using predetermined model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout #layers in neural network
from tensorflow.keras.optimizers import Adam #backpropogation
from tensorflow.keras.callbacks import EarlyStopping
import os

#set constants
IMAGES = "/home/shay/a/dani/Nutrition_AI/food-101/images"
#IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-101/images" 
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


# (.venv) bash-4.4$ python newmodel.py
# 2025-08-07 15:03:51.915978: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/thinlinc/lib64:/opt/thinlinc/lib
# 2025-08-07 15:03:51.916023: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
# Found 101000 files belonging to 101 classes.
# Using 80800 files for training.
# 2025-08-07 15:03:56.791182: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/thinlinc/lib64:/opt/thinlinc/lib
# 2025-08-07 15:03:56.791274: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
# 2025-08-07 15:03:56.791308: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ececomp4.ecn.purdue.edu): /proc/driver/nvidia/version does not exist
# 2025-08-07 15:03:56.792752: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 101000 files belonging to 101 classes.
# Using 20200 files for validation.
# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
# 9412608/9406464 [==============================] - 0s 0us/step
# 9420800/9406464 [==============================] - 0s 0us/step
# Training with frozen base model...
# Epoch 1/80
# 2025-08-07 15:04:01.907317: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
#    1/5050 [..............................] - ETA: 7:34:28 - loss: 5.5648 - accuracy: 0.00   2/5050 [..............................] - ETA: 5:18 - loss: 5.6398 - accuracy: 0.0000e   3/5050 [..............................] - ETA: 5:20 - loss: 5.4184 - accuracy: 0.0000e   4/5050 [..............................] - ETA: 5:11 - loss: 5.2293 - accuracy: 0.0000e   5/5050 [..............................] - ETA: 5:28 - loss: 5.2436 - accuracy: 0.0000e   6/5050 [..............................] - ETA: 5:24 - loss: 5.2231 - accuracy: 0.0000e   7/5050 [..............................] - ETA: 5:20 - loss: 5.2025 - accuracy: 0.0000e   8/5050 [..............................] - ETA: 5:32 - loss: 5.2418 - accuracy: 0.0000e   9/5050 [..............................] - ETA: 5:22 - loss: 5.2360 - accuracy: 0.0000e  10/5050 [..............................] - ETA: 5:20 - loss: 5.2640 - accuracy: 0.0000e  11/5050 [..............................] - ETA: 5:17 - loss: 5.2144 - accuracy: 0.0057 5050/5050 [==============================] - 341s 67ms/step - loss: 3.9886 - accuracy: 0.1146 - val_loss: 2.8799 - val_accuracy: 0.3417
# Epoch 2/80
# 5050/5050 [==============================] - 333s 66ms/step - loss: 3.1609 - accuracy: 0.2495 - val_loss: 2.4474 - val_accuracy: 0.4127
# Epoch 3/80
# 5050/5050 [==============================] - 315s 62ms/step - loss: 2.8663 - accuracy: 0.3066 - val_loss: 2.2958 - val_accuracy: 0.4398
# Epoch 4/80
# 5050/5050 [==============================] - 312s 62ms/step - loss: 2.7023 - accuracy: 0.3406 - val_loss: 2.1951 - val_accuracy: 0.4604
# Epoch 5/80
# 5050/5050 [==============================] - 310s 61ms/step - loss: 2.5823 - accuracy: 0.3664 - val_loss: 2.1467 - val_accuracy: 0.4714
# Epoch 6/80
# 5050/5050 [==============================] - 309s 61ms/step - loss: 2.4906 - accuracy: 0.3872 - val_loss: 2.0977 - val_accuracy: 0.4795
# Epoch 7/80
# 5050/5050 [==============================] - 299s 59ms/step - loss: 2.4133 - accuracy: 0.4048 - val_loss: 2.0609 - val_accuracy: 0.4865
# Epoch 8/80
# 5050/5050 [==============================] - 295s 58ms/step - loss: 2.3513 - accuracy: 0.4147 - val_loss: 2.0350 - val_accuracy: 0.4916
# Epoch 9/80
# 5050/5050 [==============================] - 271s 54ms/step - loss: 2.2975 - accuracy: 0.4290 - val_loss: 2.0144 - val_accuracy: 0.4975
# Epoch 10/80
# 5050/5050 [==============================] - 255s 50ms/step - loss: 2.2500 - accuracy: 0.4368 - val_loss: 2.0057 - val_accuracy: 0.4964
# Epoch 11/80
# 5050/5050 [==============================] - 254s 50ms/step - loss: 2.1988 - accuracy: 0.4486 - val_loss: 2.0005 - val_accuracy: 0.4985
# Epoch 12/80
# 5050/5050 [==============================] - 262s 52ms/step - loss: 2.1652 - accuracy: 0.4570 - val_loss: 1.9829 - val_accuracy: 0.5025
# Epoch 13/80
# 5050/5050 [==============================] - 271s 54ms/step - loss: 2.1242 - accuracy: 0.4640 - val_loss: 1.9910 - val_accuracy: 0.5020
# Epoch 14/80
# 5050/5050 [==============================] - 292s 58ms/step - loss: 2.0868 - accuracy: 0.4731 - val_loss: 1.9723 - val_accuracy: 0.5056
# Epoch 15/80
# 5050/5050 [==============================] - 292s 58ms/step - loss: 2.0520 - accuracy: 0.4828 - val_loss: 1.9866 - val_accuracy: 0.5032
# Epoch 16/80
# 5050/5050 [==============================] - 295s 58ms/step - loss: 2.0226 - accuracy: 0.4872 - val_loss: 1.9896 - val_accuracy: 0.5034
# Epoch 17/80
# 5050/5050 [==============================] - 295s 58ms/step - loss: 1.9854 - accuracy: 0.4909 - val_loss: 1.9712 - val_accuracy: 0.5076
# Epoch 18/80
# 5050/5050 [==============================] - 292s 58ms/step - loss: 1.9549 - accuracy: 0.5007 - val_loss: 1.9871 - val_accuracy: 0.5020
# Epoch 19/80
# 5050/5050 [==============================] - 297s 59ms/step - loss: 1.9247 - accuracy: 0.5040 - val_loss: 1.9862 - val_accuracy: 0.5059
# Epoch 20/80
# 5050/5050 [==============================] - 294s 58ms/step - loss: 1.9027 - accuracy: 0.5091 - val_loss: 1.9840 - val_accuracy: 0.5059
# Unfreezing and fine-tuning model...
# Epoch 1/20
#    1/5050 [..............................] - ETA: 4:01:21 - loss: 4.2779 - accuracy: 0.18                       2/5050 [..............................] - ETA: 21:34 - loss: 4.5842 - accuracy: 0.0938                                        5050/5050 [==============================] - 1017s 201ms/step - loss: 2.7570 - accuracy: 0.3463 - val_loss: 2.9166 - val_accuracy: 0.3207
# Epoch 2/20
# 5050/5050 [==============================] - 1021s 202ms/step - loss: 2.2595 - accuracy: 0.4425 - val_loss: 2.7047 - val_accuracy: 0.3646
# Epoch 3/20
# 5050/5050 [==============================] - 955s 189ms/step - loss: 2.0330 - accuracy: 0.4882 - val_loss: 2.6249 - val_accuracy: 0.3882
# Epoch 4/20
# 5050/5050 [==============================] - 984s 195ms/step - loss: 1.8640 - accuracy: 0.5243 - val_loss: 2.4961 - val_accuracy: 0.4132
# Epoch 5/20
# 5050/5050 [==============================] - 978s 194ms/step - loss: 1.7224 - accuracy: 0.5531 - val_loss: 2.5350 - val_accuracy: 0.4106
# Epoch 6/20
# 5050/5050 [==============================] - 987s 196ms/step - loss: 1.5987 - accuracy: 0.5795 - val_loss: 2.4832 - val_accuracy: 0.4212
# Epoch 7/20
# 5050/5050 [==============================] - 990s 196ms/step - loss: 1.4817 - accuracy: 0.6054 - val_loss: 2.6000 - val_accuracy: 0.4109
# Epoch 8/20
# 5050/5050 [==============================] - 970s 192ms/step - loss: 1.3839 - accuracy: 0.6247 - val_loss: 2.5708 - val_accuracy: 0.4177
# Epoch 9/20
# 5050/5050 [==============================] - 947s 188ms/step - loss: 1.2853 - accuracy: 0.6494 - val_loss: 2.5419 - val_accuracy: 0.4266
# /home/shay/a/dani/Nutrition_AI/.venv/lib64/python3.6/site-packages/keras/utils/generic_utils.py:497: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.
#   category=CustomMaskWarning)
# Model saved to mobilenetv2_food101_model.h5