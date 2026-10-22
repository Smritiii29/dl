
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# 1. Load and preprocess image
image_path = "cat.jpg"
image = load_img(image_path, target_size=(224, 224))
x = img_to_array(image)
img = preprocess_input(np.expand_dims(x, axis=0))

# 2. Load pre-trained CNN
model = ResNet50(weights="imagenet")

# 3. Select early, middle and deep layers
layers = ["conv1_relu", "conv2_block3_out", "conv4_block6_out"]

feature_model = Model(
    inputs=model.input,
    outputs=[model.get_layer(n).output for n in layers]
)

# 4. Extract feature maps
features = feature_model.predict(img, verbose=0)

# 5. Display original image and feature maps
plt.figure(figsize=(12, 8))
plt.subplot(4, 6, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

for i, (name, fmap) in enumerate(zip(layers, features)):
    print(name, ":", fmap.shape)

    for j in range(6):
        plt.subplot(4, 6, (i + 1) * 6 + j + 1)
        plt.imshow(fmap[0, :, :, j], cmap="viridis")
        plt.axis("off")

    plt.subplot(4, 6, (i + 1) * 6 + 1)
    plt.text(0, 0.5, name, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()

# 6. Visualize learned filters of the first convolution
filters, bias = model.get_layer("conv1_conv").get_weights()

plt.figure(figsize=(8, 8))
for i in range(16):
    f = filters[:, :, :, i]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)

    plt.subplot(4, 4, i + 1)
    plt.imshow(f)
    plt.axis("off")

plt.suptitle("First Convolution Layer Filters")
plt.show()
