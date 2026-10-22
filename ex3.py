
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset and split
path = "imagenet_subset"
train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="both", seed=1,
    image_size=(224, 224), batch_size=8
)

# 2. Load pre-trained CNN without its classifier
base = ResNet50(weights="imagenet", include_top=False,
                input_shape=(224, 224, 3))

# 3. Extract feature vectors using Global Average Pooling
model = Model(base.input, GlobalAveragePooling2D()(base.output))

def extract(ds):
    X, y = [], []
    for images, labels in ds:
        f = model.predict(preprocess_input(images.numpy()), verbose=0)
        X.extend(f)
        y.extend(labels.numpy())
    return np.array(X), np.array(y)

X_train, y_train = extract(train_ds)
X_test, y_test = extract(test_ds)

print("Feature vector shape:", X_train.shape)

# 4. Train a new classifier
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# 5. Predict unseen images
pred = clf.predict(X_test)

# 6. Evaluate
print("Accuracy:", clf.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(
    y_test, pred, target_names=train_ds.class_names,
    zero_division=0
))

# 7. Display a sample image
for images, labels in test_ds.take(1):
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(train_ds.class_names[int(labels[0])])
    plt.axis("off")
    plt.show()

