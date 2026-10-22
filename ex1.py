import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

# 1. Load dataset and split it
path = "path/to/imagenet_subset"

train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="both",
    seed=1, image_size=(224, 224), batch_size=1
)

# 2. Load pre-trained CNN
model = ResNet50(weights="imagenet")

# 3. Get one test image and its original label
x, y = next(iter(test_ds))
original_label = test_ds.class_names[int(y[0])]

print("Original image shape:", x.shape)
print("Original label:", original_label)

# 4. Preprocess image
img = preprocess_input(x.numpy().copy())
print("Preprocessed shape:", img.shape)

# 5. Predict top-5 classes
pred = model.predict(img, verbose=0)
top5 = decode_predictions(pred, top=5)[0]

for _, label, score in top5:
    print(label, ":", round(float(score) * 100, 2), "%")

print("Top-1 prediction:", top5[0][1])
print("Label comparison:", original_label, "vs", top5[0][1])

# 6. Extract an intermediate feature map
feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer("conv1_relu").output
)
features = feature_model.predict(img, verbose=0)
print("Feature map shape:", features.shape)

# 7. Visualize image and feature map
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(x[0].numpy().astype("uint8"))
plt.title("Original: " + original_label)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(features[0, :, :, 0], cmap="viridis")
plt.title("CNN Feature Map")
plt.axis("off")
plt.show()
