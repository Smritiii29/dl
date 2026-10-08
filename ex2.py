import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# 1. Load ImageNet-derived dataset automatically
ds = tfds.load(
    "imagenette/160px-v2",
    split="train",
    as_supervised=True
)

# 2. Get one image
img, label = next(iter(ds))

# 3. Resize and preprocess
img = tf.image.resize(img, (224, 224))
x = np.expand_dims(img.numpy(), 0)
x = preprocess_input(x)

# 4. Load pretrained VGG16
model = VGG16(weights="imagenet")

# 5. Select 3 convolutional layers
layers = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv1"
]

feature_model = Model(
    model.input,
    [model.get_layer(l).output for l in layers]
)

features = feature_model.predict(x, verbose=0)

# 6. Display feature maps
for name, feature in zip(layers, features):

    print(name, ":", feature.shape)

    plt.imshow(feature[0, :, :, 0], cmap="gray")
    plt.title(name + " Feature Map")
    plt.axis("off")
    plt.show()

# 7. Display learned filters
filters = model.get_layer("block1_conv1").get_weights()[0]

print("Filter shape:", filters.shape)

for i in range(6):
    plt.imshow(filters[:, :, 0, i], cmap="gray")
    plt.title("Filter " + str(i))
    plt.axis("off")
    plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Model

# model = VGG16(weights="imagenet")

# img = image.load_img("cat.jpg", target_size=(224,224))
# x = np.expand_dims(image.img_to_array(img), 0)
# x = preprocess_input(x)

# layers = ["block1_conv1", "block3_conv1", "block5_conv1"]

# fm = Model(model.input,
#            [model.get_layer(l).output for l in layers])

# features = fm.predict(x, verbose=0)

# for l, f in zip(layers, features):
#     print(l, ":", f.shape)
#     plt.imshow(f[0,:,:,0], cmap="gray")
#     plt.title(l)
#     plt.axis("off")
#     plt.show()

# filters = model.get_layer("block1_conv1").get_weights()[0]
# print("Filter shape:", filters.shape)

# for i in range(6):
#     plt.imshow(filters[:,:,0,i], cmap="gray")
#     plt.title("Filter " + str(i+1))
#     plt.axis("off")
#     plt.show()