# source /home/shay/a/dani/Nutrition_AI/.venv/bin/activate

#improvements to implement for v3:
#use new framework -- efficientnetv2S - done
#change learning rates for pre and post freezing -- try using cosine decay schedule or AdamW optimizer
#change image sizes to 224x224 & 256x256 to get better feature extraction - done
#we can clearly see a large discrepency between training and validation accuracy -- try more extreme feature manipulation so more rotation/brightness etc. - done
#change regularization layers (dropout maybe like ~0.3) to extract more features - done


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S #using predetermined model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout #layers in neural network
from tensorflow.keras.optimizers import Adam #backpropogation
from tensorflow.keras.callbacks import EarlyStopping

#set constants
IMAGES = "/home/shay/a/dani/Nutrition_AI/food-101/images" #use this path when on linux
# IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-101/images"  #use this path when on mac
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8  # reduced from 32 to expedite training on CPU
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
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomBrightness(factor=0.25)
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

# remove caching as it leads to memory exhaustion
# train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #caches images so faster to retrieve, shuffles a buffer to randomly sample, and prefetches data while the model is training in a previous batch
# val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

#using the EfficientNetV2M model
base_model = EfficientNetV2S(
    input_shape=(256, 256, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.35)(x) #changed dropout to 0.35 to hopefully try and pick out more niche features -- can be bad if we have overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.35)(x)
outputs = Dense(TOTAL_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

#compile and train the model for the frozen dataset
model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

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
model_path = "EfficientNetV2S_food101_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")

