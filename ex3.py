import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
ds = tfds.load(
    "imagenette/160px-v2",
    split="train",
    as_supervised=True
)

# 2. Collect 50 images from each of first 5 classes
images = []
labels = []
count = [0] * 5

for img, label in tfds.as_numpy(ds):
    label = int(label)

    if label < 5 and count[label] < 50:
        img = tf.image.resize(img, (224, 224)).numpy()
        images.append(img)
        labels.append(label)
        count[label] += 1

    if all(c == 50 for c in count):
        break

X = preprocess_input(np.array(images))
y = np.array(labels)

print("Dataset:", X.shape)
print("Labels:", y.shape)

# 3. Load pretrained VGG16 without final classifier
cnn = VGG16(weights="imagenet", include_top=False)

# 4. Feature extractor
extractor = Model(
    cnn.input,
    GlobalAveragePooling2D()(cnn.output)
)

X_features = extractor.predict(X, verbose=0)

print("Feature vector shape:", X_features.shape)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. New classification layer
classifier = tf.keras.Sequential([
    Dense(5, activation="softmax")
])

classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 7. Train
classifier.fit(
    X_train,
    y_train,
    epochs=10,
    verbose=1
)

# 8. Test
pred = np.argmax(
    classifier.predict(X_test, verbose=0),
    axis=1
)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# path = "imagenet_subset"

# data = ImageDataGenerator(
#     preprocessing_function=preprocess_input
# ).flow_from_directory(
#     path, target_size=(224,224), batch_size=10,
#     class_mode="sparse", shuffle=False
# )

# cnn = VGG16(weights="imagenet")

# extractor = Model(
#     cnn.input,
#     GlobalAveragePooling2D()(cnn.get_layer("block5_pool").output)
# )

# X = extractor.predict(data, verbose=0)
# y = data.classes

# np.save("features.npy", X)
# np.save("labels.npy", y)

# print("Feature vector:", X.shape)

# Xtr, Xte, ytr, yte = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# clf = tf.keras.Sequential([
#     Dense(len(data.class_indices), activation="softmax")
# ])

# clf.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# clf.fit(Xtr, ytr, epochs=20, verbose=0)

# p = np.argmax(clf.predict(Xte, verbose=0), axis=1)

# print("Accuracy:", accuracy_score(yte, p))
# print(classification_report(yte, p))
# print("Confusion Matrix:")
# print(confusion_matrix(yte, p))