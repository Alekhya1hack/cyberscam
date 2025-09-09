import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ------------------ Data Setup ------------------
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "qr_dataset",           # folder name
    target_size=(128, 128), # resize images
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "qr_dataset",
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# ------------------ Model ------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ------------------ Training ------------------
model.fit(train_generator, validation_data=val_generator, epochs=5)

# ------------------ Save model ------------------
model.save("cnn_model.h5")

print("✅ Training complete, cnn_model.h5 saved!")
