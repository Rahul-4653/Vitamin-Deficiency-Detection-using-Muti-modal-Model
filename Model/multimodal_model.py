import os
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================
# PARAMETERS
# =============================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

skin_dir = "../Dataset/Skin_Images"
nail_dir = "../Dataset/Nail_Images"
symptom_file = "../Dataset/Updated_Symptoms.xlsx"


# =============================
# CHECK PATHS
# =============================

if not os.path.exists(skin_dir):
    raise FileNotFoundError(f"Skin dataset not found: {skin_dir}")

if not os.path.exists(nail_dir):
    raise FileNotFoundError(f"Nail dataset not found: {nail_dir}")

if not os.path.exists(symptom_file):
    raise FileNotFoundError(f"Excel file not found: {symptom_file}")


# =============================
# DATASET CLEANING FUNCTION
# =============================

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

def clean_dataset(folder):

    print(f"\nScanning dataset: {folder}")

    for root, dirs, files in os.walk(folder):

        for file in files:

            path = os.path.join(root, file)

            if not file.lower().endswith(VALID_EXTENSIONS):
                print("Removing non-image:", path)
                os.remove(path)
                continue

            try:
                img = Image.open(path)
                img.verify()
            except Exception:
                print("Removing corrupted image:", path)
                os.remove(path)


clean_dataset(skin_dir)
clean_dataset(nail_dir)


# =============================
# IMAGE GENERATOR
# =============================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)


# Skin dataset
skin_train = datagen.flow_from_directory(
    skin_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

skin_val = datagen.flow_from_directory(
    skin_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


# Nail dataset
nail_train = datagen.flow_from_directory(
    nail_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

nail_val = datagen.flow_from_directory(
    nail_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)


# =============================
# MODEL BUILDER
# =============================

def build_model(num_classes):

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =============================
# SKIN MODEL TRAINING
# =============================

print("\nTraining skin model...")

skin_model = build_model(skin_train.num_classes)

skin_model.fit(
    skin_train,
    validation_data=skin_val,
    epochs=EPOCHS
)

skin_model.save("skin_model.h5")


# =============================
# CLEAR SESSION
# =============================

tf.keras.backend.clear_session()


# =============================
# NAIL MODEL TRAINING
# =============================

print("\nTraining nail model...")

nail_model = build_model(nail_train.num_classes)

nail_model.fit(
    nail_train,
    validation_data=nail_val,
    epochs=EPOCHS
)

nail_model.save("nail_model.h5")


# =============================
# SYMPTOM MODEL
# =============================

print("\nTraining symptom model...")

symptoms = pd.read_excel(symptom_file, engine="openpyxl")

# Clean column names
symptoms.columns = symptoms.columns.str.strip().str.lower()

print("Columns detected:", symptoms.columns)


# =============================
# LABEL COLUMN FIX
# =============================

label_column = "b12_status"

if label_column not in symptoms.columns:
    raise ValueError(
        f"Column '{label_column}' not found in Excel file"
    )

X = symptoms.drop(label_column, axis=1)
y = symptoms[label_column]


# =============================
# ENCODE LABELS
# =============================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Label classes:", encoder.classes_)


# =============================
# TRAIN TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =============================
# NORMALIZE FEATURES
# =============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =============================
# SYMPTOM MODEL
# =============================

symptom_model = tf.keras.Sequential([

    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(len(np.unique(y)), activation="softmax")

])


symptom_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


symptom_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)


symptom_model.save("symptom_model.h5")


print("\nTraining completed successfully!")
print("Models saved:")
print("skin_model.h5")
print("nail_model.h5")
print("symptom_model.h5")
