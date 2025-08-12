import tensorflow as tf
from tensorflow.keras.applications import ResNet50 #this is the model that we will use (predetermined layers and neurons)
from tensorflow.keras.preprocessing.image import ImageDataGenerator #helps us get image from disk
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout #helps connectivity
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam #this is for backpropogation
from tensorflow.keras.callbacks import EarlyStopping

#set constants
IMAGES = "/Users/raunaksmac/Desktop/Nutrition AI/food-101/images"
FREEZE_EPOCHS = 30 #this is when resnet is frozen
UNFREEZE_EPOCHS = 5
BATCH_SIZE = 32
TOTAL_CLASSES = 101#azerbaijan intelligence 
IMAGE_SIZE = (224, 224) #this is the standard size for resnet

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#import data and rescale to 0-1
datagen = ImageDataGenerator(
    rescale=1./255, #scales to 0-1
    horizontal_flip=True, #this is to learn features better
    validation_split=0.2, #20% of data used for validation
    rotation_range=30, #rotates image between -90 and 90
    zoom_range=[0.8, 1.2]
)

#split into training and validation sets
training = datagen.flow_from_directory(
    IMAGES,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation = datagen.flow_from_directory(
    IMAGES,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation'
)
#build the model that is frozen
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

#Use a Functional API to allow for extraction of more nonlinear features and complelx combinations
tensor = base_model.output #outputs a feature map of (7, 7, 2048) -- (height, width, channels)
one_hot_encoded = GlobalAveragePooling2D()(tensor) #reduces dimensionality by utilizing 1-hot encoding -- similar to flatten, but better for large models as it uses average instaed of taking all features
first_dense = Dense(512, activation="relu")(one_hot_encoded)
drop_tensors = Dropout(rate=0.5)(first_dense) #selects only 50% of tensors to be turned on
second_dense = Dense(256, activation="relu")(drop_tensors)
drop_tensors_again = Dropout(rate=0.5)(second_dense) 
out = Dense(101, activation="softmax")(drop_tensors_again)
full_model = Model(inputs=base_model.input, outputs=out)

#compile + train the model without imagenet for resnet50
full_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(1e-4),
    metrics=["accuracy"]
)

full_model.fit(
    training,
    validation_data=validation,
    epochs=FREEZE_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

#active food components of imagenet for resnet50
print("Fine-tuning entire model...")
base_model.trainable = True
full_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(1e-5),
    metrics=["accuracy"]
)

full_model.fit(
    training, 
    validation_data=validation,
    epochs=UNFREEZE_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

#save final model
full_model.save('resnet50_food101_model.h5')
print("Model saved as resnet50_food101_model.h5")


#val accuracy is for validation set, so if val_accuracy decreases and training accuracy increases, 
# then we overfit to the trianing dat