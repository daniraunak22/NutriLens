# # source /home/shay/a/dani/Nutrition_AI/.venv/bin/activate

# #improvements to implement for v3:
# #use new framework -- efficientnetv2S - done
# #change learning rates for pre and post freezing -- try using cosine decay schedule or AdamW optimizer
# #change image sizes to 224x224 & 256x256 to get better feature extraction - done
# #we can clearly see a large discrepency between training and validation accuracy -- try more extreme feature manipulation so more rotation/brightness etc. - done
# #change regularization layers (dropout maybe like ~0.3) to extract more features - done


# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetV2S #using predetermined model
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input #layers in neural network
# from tensorflow.keras.optimizers import AdamW #backpropogation while minimizing overfitting
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import mixed_precision

# mixed_precision.set_global_policy('mixed_float16') #performs all calculations in 16bits and then the last dense layer outputs 32 bits

# print("Available GPUs:", tf.config.list_physical_devices("GPU")) #indicates we can use the GPU on M3 Max chip
# tf.debugging.set_log_device_placement(True) #logs if we train on CPU or GPU
# #uses the GPU if it is available else it will use the CPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# #set constants
# # IMAGES = "/home/shay/a/dani/Nutrition_AI/food-101/images" #use this path when on linux
# IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-101/images"  #use this path when on mac
# IMAGE_SIZE = (256, 256) #alternate size can also be (224x224)
# BATCH_SIZE = 64  # increase batch size from 8 to 64 as training is shifted to GPU, not CPU
# TOTAL_CLASSES = 101
# FREEZE_EPOCHS = 40
# UNFREEZE_EPOCHS = 20
# SEED = 123

# #autotune data to optimize parallel processing
# AUTOTUNE = tf.data.AUTOTUNE

# #load data
# train_data = tf.keras.utils.image_dataset_from_directory(
#     IMAGES,
#     validation_split=0.1, #because we are training on 100,000+ images, I will use 10% for validation and not 20%
#     subset="training",
#     seed=SEED,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="categorical",
#     shuffle=True
# )

# val_data = tf.keras.utils.image_dataset_from_directory(
#     IMAGES,
#     validation_split=0.1,
#     subset="validation",
#     seed=SEED,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="categorical",
# )

# #because we automatically load data in, then use data augmentation to randomly flip, rotate, and zoom images for better training
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.25),
#     tf.keras.layers.RandomZoom(0.1),
#     tf.keras.layers.RandomBrightness(0.25)
# ], name="data_augmentation")

# #scales color of images to 0-1 and implements the flipping/rotation/zoom (normalization)
# def preprocess_train(image, label):
#     image = tf.cast(image, tf.float32) / 255.0
#     return image, label

# def preprocess_val(image, label):
#     image = tf.cast(image, tf.float32) / 255.0
#     return image, label

# # remove caching as it leads to memory exhaustion
# train_data = train_data.prefetch(buffer_size=AUTOTUNE) #prefetches data while the model is training in a previous batch
# val_data = val_data.prefetch(AUTOTUNE)

# #using the EfficientNetV2M model
# inputs = Input(shape=(256, 256, 3))
# x = tf.keras.layers.Rescaling(1./255)(inputs)  # normalize first
# x = data_augmentation(x)  # augment after normalization
# base_model = EfficientNetV2S(include_top=False, 
#                              weights='imagenet', 
#                              input_tensor=x)
# base_model.trainable = False

# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.35)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.35)(x)
# outputs = Dense(TOTAL_CLASSES, activation='softmax', dtype='float32')(x)

# model = Model(inputs=inputs, outputs=outputs)
# #compile and train the model for the frozen dataset
# optimizer_frozen = AdamW(learning_rate=1e-4, weight_decay=1e-5)
# model.compile(
#     optimizer=optimizer_frozen,
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# print("Training with frozen base model...")
# model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=FREEZE_EPOCHS,
#     callbacks=[early_stop]
# )

# #compile and train the model for the unfrozen dataset
# print("Unfreezing and fine-tuning model...")
# base_model.trainable = True

# optimizer_unfreeze = AdamW(learning_rate=1e-5, weight_decay=1e-5)
# model.compile(
#     optimizer=optimizer_unfreeze,
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=UNFREEZE_EPOCHS,
#     callbacks=[early_stop]
# )

# #saving model
# model_path = "EfficientNetV2S_food101_model.h5"
# model.save(model_path)
# print(f"Model saved to {model_path}")

# source /home/shay/a/dani/Nutrition_AI/.venv/bin/activate

# source /home/shay/a/dani/Nutrition_AI/.venv/bin/activate

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

print("Available GPUs:", tf.config.list_physical_devices("GPU"))
tf.debugging.set_log_device_placement(True)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
# IMAGES = "/home/shay/a/dani/Nutrition_AI/food-101/images" # linux
IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-101/images"  # mac
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32  # Reduced from 64 to help with stability
TOTAL_CLASSES = 101
FREEZE_EPOCHS = 15  # Reduced for better convergence
UNFREEZE_EPOCHS = 25
SEED = 123

AUTOTUNE = tf.data.AUTOTUNE

# Load data with proper validation split
train_data = tf.keras.utils.image_dataset_from_directory(
    IMAGES,
    validation_split=0.2,  # Using 20% for better validation
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    IMAGES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

# Print class names to verify data loading
class_names = train_data.class_names
print(f"Found {len(class_names)} classes: {class_names[:10]}...")  # Show first 10

# Data augmentation (applied only during training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),  # Reduced rotation
    tf.keras.layers.RandomZoom(0.15),     # Slightly increased zoom
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),  # Added contrast augmentation
], name="data_augmentation")

# Preprocessing functions - FIXED: Only normalize once
def preprocess_train(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = data_augmentation(image, training=True)  # Apply augmentation only to training
    return image, label

def preprocess_val(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing
train_data = train_data.map(preprocess_train, num_parallel_calls=AUTOTUNE)
val_data = val_data.map(preprocess_val, num_parallel_calls=AUTOTUNE)

# Optimize data pipeline
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# Create model - FIXED: Removed double normalization
def create_model():
    inputs = Input(shape=(256, 256, 3))
    
    # Create base model without input preprocessing
    base_model = EfficientNetV2S(
        include_top=False, 
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    
    # Pass inputs directly to base model (preprocessing done in data pipeline)
    x = base_model(inputs, training=False)  # Set training=False for frozen training
    
    # Add custom head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)  # Added batch normalization
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)  # Slightly reduced dropout
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(TOTAL_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    base_model.trainable = False
    
    return model, base_model

# Create model
model, base_model = create_model()

# Print model summary
print("Model Summary:")
model.summary()

# Compile model for frozen training with higher learning rate
optimizer_frozen = AdamW(learning_rate=1e-3, weight_decay=1e-4)  # Increased LR
model.compile(
    optimizer=optimizer_frozen,
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Removed top_5_accuracy for compatibility
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',  # Changed to monitor accuracy
    patience=7, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

print("Training with frozen base model...")
history_frozen = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FREEZE_EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# # Fine-tuning phase
# print("Unfreezing and fine-tuning model...")
# base_model.trainable = True

# # Freeze early layers, only fine-tune later layers
# for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
#     layer.trainable = False

# # Use lower learning rate for fine-tuning
# optimizer_unfreeze = AdamW(learning_rate=1e-5, weight_decay=1e-4)
# model.compile(
#     optimizer=optimizer_unfreeze,
#     loss='categorical_crossentropy',
#     metrics=['accuracy']  # Removed top_5_accuracy for compatibility
# )

# # Reset callbacks for fine-tuning
# early_stop_finetune = EarlyStopping(
#     monitor='val_accuracy',
#     patience=10,
#     restore_best_weights=True,
#     verbose=1
# )

# reduce_lr_finetune = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.3,
#     patience=4,
#     min_lr=1e-8,
#     verbose=1
# )

# history_unfreeze = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=UNFREEZE_EPOCHS,
#     callbacks=[early_stop_finetune, reduce_lr_finetune],
#     verbose=1
# )

# Save model
model_path = "EfficientNetV2S_food101_model_improved.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

# Print final results
print(f"\nFinal Training Accuracy: {max(history_frozen.history['accuracy']):.4f}")
print(f"Final Validation Accuracy: {max(history_frozen.history['val_accuracy']):.4f}")

# Final Training Accuracy: 0.0123
# Final Validation Accuracy: 0.0178