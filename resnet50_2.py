import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === CONSTANTS ===
IMAGES = "/home/shay/a/dani/Nutrition AI/food-101/images"
FREEZE_EPOCHS = 60
UNFREEZE_EPOCHS = 20
BATCH_SIZE = 32
TOTAL_CLASSES = 101
IMAGE_SIZE = (224, 224)

# === CALLBACKS ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === DATA AUGMENTATION ===
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    zoom_range=[0.8, 1.2],
    validation_split=0.2
)

# === DATA LOADERS ===
train_ds = datagen.flow_from_directory(
    IMAGES,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_ds = datagen.flow_from_directory(
    IMAGES,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# === BUILD MODEL ===
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze during initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(TOTAL_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === COMPILE AND TRAIN (FROZEN BASE) ===
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[Phase 1] Training with frozen base...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FREEZE_EPOCHS,
    callbacks=[early_stop],
    steps_per_epoch=train_ds.samples // BATCH_SIZE,
    validation_steps=val_ds.samples // BATCH_SIZE
)

# === UNFREEZE BASE MODEL FOR FINE-TUNING ===
print("[Phase 2] Unfreezing base model and fine-tuning...")
base_model.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower LR to avoid destroying pretrained weights
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=UNFREEZE_EPOCHS,
    callbacks=[early_stop],
    steps_per_epoch=train_ds.samples // BATCH_SIZE,
    validation_steps=val_ds.samples // BATCH_SIZE
)

# === SAVE FINAL MODEL ===
model.save('resnet50_food101_model.h5')
print("✅ Model saved as resnet50_food101_model.h5")
