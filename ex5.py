import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
ds = tfds.load(
    "imagenette/160px-v2",
    split="train",
    as_supervised=True
)

# 2. Take 5 classes, 100 images each
images = []
labels = []
count = [0] * 5

for img, label in tfds.as_numpy(ds):
    label = int(label)

    if label < 5 and count[label] < 100:
        img = tf.image.resize(img, (32, 32)).numpy()
        images.append(img)
        labels.append(label)
        count[label] += 1

    if all(c == 100 for c in count):
        break

X = np.array(images).astype("float32") / 127.5 - 1
y = tf.keras.utils.to_categorical(labels, 5)

# 3. Generator
noise = tf.keras.Input(shape=(100,))
text = tf.keras.Input(shape=(5,))

x = tf.keras.layers.Concatenate()([noise, text])
x = tf.keras.layers.Dense(8 * 8 * 128, activation="relu")(x)
x = tf.keras.layers.Reshape((8, 8, 128))(x)
x = tf.keras.layers.Conv2DTranspose(
    64, 4, strides=2, padding="same", activation="relu"
)(x)
x = tf.keras.layers.Conv2DTranspose(
    32, 4, strides=2, padding="same", activation="relu"
)(x)
output = tf.keras.layers.Conv2D(
    3, 3, padding="same", activation="tanh"
)(x)

generator = tf.keras.Model([noise, text], output)

# 4. Discriminator
image = tf.keras.Input(shape=(32, 32, 3))
text = tf.keras.Input(shape=(5,))

x = tf.keras.layers.Conv2D(
    32, 4, strides=2, padding="same"
)(image)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(
    64, 4, strides=2, padding="same"
)(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Concatenate()([x, text])

output = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

discriminator = tf.keras.Model(
    [image, text],
    output
)

# 5. Compile discriminator
discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# 6. Combined GAN
discriminator.trainable = False

z = tf.keras.Input(shape=(100,))
t = tf.keras.Input(shape=(5,))

fake = generator([z, t])
valid = discriminator([fake, t])

gan = tf.keras.Model([z, t], valid)

gan.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# 7. Train
for epoch in range(20):

    noise = np.random.normal(0, 1, (32, 100))
    idx = np.random.randint(0, len(X), 32)

    real_images = X[idx]
    real_text = y[idx]

    fake_images = generator.predict(
        [noise, real_text],
        verbose=0
    )

    discriminator.trainable = True

    discriminator.train_on_batch(
        [real_images, real_text],
        np.ones((32, 1))
    )

    discriminator.train_on_batch(
        [fake_images, real_text],
        np.zeros((32, 1))
    )

    discriminator.trainable = False

    gan.train_on_batch(
        [noise, real_text],
        np.ones((32, 1))
    )

    if epoch % 5 == 0:
        print("Epoch:", epoch)

# 8. Generate an image
noise = np.random.normal(0, 1, (1, 100))

text = np.zeros((1, 5))
text[0, 0] = 1       # first class

generated = generator.predict(
    [noise, text],
    verbose=0
)

generated = (generated[0] + 1) / 2

plt.imshow(generated)
plt.title("Generated Image")
plt.axis("off")
plt.show()
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt

# path = "imagenet_subset"
# classes = os.listdir(path)
# N = len(classes)

# X, Y = [], []

# for i, c in enumerate(classes):
#     for f in os.listdir(os.path.join(path,c)):
#         try:
#             img = load_img(
#                 os.path.join(path,c,f),
#                 target_size=(32,32)
#             )
#             X.append(img_to_array(img)/255.)
#             Y.append(i)
#         except:
#             pass

# X = np.array(X)
# Y = tf.keras.utils.to_categorical(Y,N)

# # Generator
# noise = layers.Input((100,))
# text = layers.Input((N,))
# z = layers.Concatenate()([noise,text])
# z = layers.Dense(8*8*128, activation="relu")(z)
# z = layers.Reshape((8,8,128))(z)
# z = layers.Conv2DTranspose(64,4,2,"same",activation="relu")(z)
# out = layers.Conv2DTranspose(3,4,2,"same",activation="sigmoid")(z)

# G = tf.keras.Model([noise,text],out)

# # Discriminator
# img = layers.Input((32,32,3))
# text = layers.Input((N,))
# d = layers.Conv2D(64,4,2,"same",activation="relu")(img)
# d = layers.Flatten()(d)
# t = layers.Dense(64,activation="relu")(text)
# d = layers.Concatenate()([d,t])
# d = layers.Dense(1,activation="sigmoid")(d)

# D = tf.keras.Model([img,text],d)
# D.compile(optimizer="adam",loss="binary_crossentropy")

# # Conditional GAN
# noise = layers.Input((100,))
# text = layers.Input((N,))
# fake = G([noise,text])

# D.trainable = False
# GAN = tf.keras.Model([noise,text],D([fake,text]))
# GAN.compile(optimizer="adam",loss="binary_crossentropy")
# D.trainable = True

# # Train
# for epoch in range(100):
#     z = np.random.normal(size=(len(X),100))
#     fake = G.predict([z,Y],verbose=0)

#     D.train_on_batch([X,Y],np.ones((len(X),1)))
#     D.train_on_batch([fake,Y],np.zeros((len(X),1)))

#     z = np.random.normal(size=(len(X),100))
#     GAN.train_on_batch([z,Y],np.ones((len(X),1)))

# # Generate
# i = classes.index("tiger")
# z = np.random.normal(size=(1,100))
# text = tf.keras.utils.to_categorical([i],N)

# generated = G.predict([z,text],verbose=0)[0]

# plt.imshow(generated)
# plt.title("A photo of a tiger")
# plt.axis("off")
# plt.show()