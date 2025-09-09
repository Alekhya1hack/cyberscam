#train_cnn.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import argparse

def build_model(input_shape=(224,224,3)):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(base.input, out)
    return model

def main(data_dir, out_model, epochs=10, batch=16):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(train_dir, target_size=(224,224), batch_size=batch, class_mode='binary')
    val_flow = val_gen.flow_from_directory(val_dir, target_size=(224,224), batch_size=batch, class_mode='binary')

    model = build_model((224,224,3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    model.fit(train_flow, validation_data=val_flow, epochs=epochs)
    model.save(out_model)
    print("Saved model to", out_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_crops")
    parser.add_argument("--out", default="models/cnn_model.h5")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.data_dir, args.out, epochs=args.epochs)
