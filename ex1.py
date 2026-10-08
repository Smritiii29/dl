# import os, random
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Model

# model = VGG16(weights="imagenet")

# path = "imagenet_subset"
# label = random.choice(os.listdir(path))
# file = random.choice(os.listdir(os.path.join(path, label)))

# img = image.load_img(os.path.join(path, label, file), target_size=(224,224))
# x = np.expand_dims(image.img_to_array(img), 0)
# x = preprocess_input(x)

# print("Input shape:", x.shape)

# pred = model.predict(x, verbose=0)
# top5 = decode_predictions(pred, top=5)[0]

# print("Original label:", label)
# print("\nTop-5 predictions:")
# for i, (_, name, p) in enumerate(top5):
#     print(i+1, name, round(p*100,2), "%")

# print("Predicted label:", top5[0][1])

# feature_model = Model(model.input,
#                       model.get_layer("block5_conv3").output)
# features = feature_model.predict(x, verbose=0)

# print("Intermediate shape:", features.shape)

# plt.imshow(img)
# plt.title(label)
# plt.axis("off")
# plt.show()

# plt.imshow(features[0,:,:,0], cmap="gray")
# plt.title("Intermediate Feature Map")
# plt.axis("off")
# plt.show()
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

# 1. Load dataset automatically
ds, info = tfds.load(
    "imagenette/160px-v2",
    split="train",
    as_supervised=True,
    with_info=True
)

class_names = info.features["label"].names

# 2. Take one image
img, label = next(iter(ds))

print("Original label:", class_names[label.numpy()])
print("Original shape:", img.shape)

# 3. Resize
img = tf.image.resize(img, (224, 224))

# 4. Convert to array and preprocess
x = np.expand_dims(img.numpy(), axis=0)
x = preprocess_input(x)

print("Preprocessed shape:", x.shape)

# 5. Display image
plt.imshow(img.numpy().astype("uint8"))
plt.title("Original: " + class_names[label.numpy()])
plt.axis("off")
plt.show()

# 6. Load pretrained VGG16
model = VGG16(weights="imagenet")

# 7. Predict
pred = model.predict(x, verbose=0)

# 8. Top-5 predictions
top5 = decode_predictions(pred, top=5)[0]

print("\nTop-5 predictions:")
for _, name, probability in top5:
    print(name, ":", round(probability * 100, 2), "%")

# 9. Intermediate representation
feature_model = Model(
    model.input,
    model.get_layer("block5_conv3").output
)

features = feature_model.predict(x, verbose=0)

print("\nIntermediate feature shape:", features.shape)