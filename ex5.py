
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Load class-folder images
ds = tf.keras.utils.image_dataset_from_directory(
    "path/to/ImageNet", image_size=(32, 32), batch_size=32
)
names = ds.class_names
n = len(names)
ds = ds.map(lambda x, y: (x / 127.5 - 1, tf.one_hot(y, n)))

# 2. Conditional Generator: noise + text/class condition -> image
z = layers.Input((100,))
c = layers.Input((n,))
x = layers.Concatenate()([z, c])
x = layers.Dense(8 * 8 * 128, activation="relu")(x)
x = layers.Reshape((8, 8, 128))(x)
x = layers.Conv2DTranspose(64, 4, 2, padding="same",
                           activation="relu")(x)
out = layers.Conv2DTranspose(3, 4, 2, padding="same",
                             activation="tanh")(x)
G = tf.keras.Model([z, c], out)

# 3. Conditional Discriminator: image + class condition -> real/fake
img = layers.Input((32, 32, 3))
cond = layers.Input((n,))
cmap = layers.Dense(32 * 32, activation="relu")(cond)
cmap = layers.Reshape((32, 32, 1))(cmap)
x = layers.Concatenate()([img, cmap])
x = layers.Conv2D(64, 4, 2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Flatten()(x)
out = layers.Dense(1)(x)
D = tf.keras.Model([img, cond], out)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
go = tf.keras.optimizers.Adam(0.0002)
do = tf.keras.optimizers.Adam(0.0002)

# 4. Adversarial training
for epoch in range(5):
    for real, c in ds:
        b = tf.shape(real)[0]
        noise = tf.random.normal((b, 100))

        with tf.GradientTape() as tape:
            fake = G([noise, c], training=True)
            loss_d = bce(tf.ones((b, 1)), D([real, c])) + \
                     bce(tf.zeros((b, 1)), D([fake, c]))
        do.apply_gradients(zip(
            tape.gradient(loss_d, D.trainable_variables),
            D.trainable_variables))

        with tf.GradientTape() as tape:
            fake = G([noise, c], training=True)
            loss_g = bce(tf.ones((b, 1)), D([fake, c]))
        go.apply_gradients(zip(
            tape.gradient(loss_g, G.trainable_variables),
            G.trainable_variables))

    print("Epoch:", epoch + 1, "D:", float(loss_d),
          "G:", float(loss_g))

# 5. Generate an image for a selected class
class_id = names.index(names[0])
condition = tf.one_hot([class_id], n)
generated = G([tf.random.normal((1, 100)), condition],
              training=False)[0]
plt.imshow((generated.numpy() + 1) / 2)
plt.title(names[class_id])
plt.axis("off")
plt.show()

# GAN stands for Generative Adversarial Network. It contains two networks competing with each other:
# - Generator (G): creates synthetic examples.
# - Discriminator (D): judges whether an image is real or generated.
# The generator improves by trying to fool the discriminator. The discriminator improves by distinguishing real and fake images.
# Generator
# Receives noise and condition, then produces an image. Dense and transposed-convolution layers can progressively expand a vector into an image.
# Discriminator
# Receives an image and the matching condition and produces a real/fake score. It should learn to identify real images as real and generated images as fake.
# Adversarial training
# A typical training iteration:
# 1. Sample real images and class conditions.
# 2. Sample random noise.
# 3. Generate fake images.
# 4. Train D to score real images as real and fake images as fake.
# 5. Train G so D classifies generated images as real.
# 6. Repeat over batches and epochs.