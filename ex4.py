import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load dataset
ds, info = tfds.load(
    "imagenette/160px-v2",
    split="train",
    as_supervised=True,
    with_info=True
)

class_names = info.features["label"].names

# 2. Collect 20 images from first 5 classes
images = []
labels = []
count = [0] * 5

for img, label in tfds.as_numpy(ds):
    label = int(label)

    if label < 5 and count[label] < 20:
        images.append(
            tf.image.resize(img, (224, 224)).numpy()
        )
        labels.append(label)
        count[label] += 1

    if all(c == 20 for c in count):
        break

X = preprocess_input(np.array(images))

# 3. CNN feature extractor
cnn = VGG16(weights="imagenet", include_top=False)

extractor = Model(
    cnn.input,
    tf.keras.layers.GlobalAveragePooling2D()(cnn.output)
)

features = extractor.predict(X, verbose=0)

# 4. Create captions from class labels
captions = [
    "<start> a photo of a " + class_names[label] + " <end>"
    for label in labels
]

print("Example caption:")
print(captions[0])

# 5. Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

sequences = tokenizer.texts_to_sequences(captions)
max_len = max(len(s) for s in sequences)

X_seq = pad_sequences(sequences, maxlen=max_len, padding="post")

# Input = words except last word
X_words = X_seq[:, :-1]

# Target = next words
Y_words = X_seq[:, 1:]

vocab_size = len(tokenizer.word_index) + 1

# Convert target to one-hot
Y = tf.keras.utils.to_categorical(
    Y_words,
    num_classes=vocab_size
)

# 6. CNN + LSTM model
image_input = Input(shape=(512,))
text_input = Input(shape=(max_len - 1,))

h = Dense(128, activation="tanh")(image_input)
c = Dense(128, activation="tanh")(image_input)

x = Embedding(
    vocab_size,
    128,
    mask_zero=True
)(text_input)

x = LSTM(
    128,
    return_sequences=True
)(x, initial_state=[h, c])

output = Dense(
    vocab_size,
    activation="softmax"
)(x)

model = Model(
    [image_input, text_input],
    output
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)

# 7. Train
model.fit(
    [features, X_words],
    Y,
    epochs=10,
    verbose=1
)

# 8. Generate a caption
image_id = 0
words = ["<start>"]

for i in range(max_len - 1):

    seq = tokenizer.texts_to_sequences(
        [" ".join(words)]
    )

    seq = pad_sequences(
        seq,
        maxlen=max_len - 1,
        padding="post"
    )

    pred = model.predict(
        [features[image_id:image_id+1], seq],
        verbose=0
    )

    next_word = np.argmax(pred[0, i])

    if next_word == 0:
        break

    word = tokenizer.index_word.get(next_word, "")

    if word == "<end>":
        break

    words.append(word)

print("\nGenerated caption:")
print(" ".join(words[1:]))
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, RepeatVector, Concatenate, TimeDistributed
# from tensorflow.keras.models import Model

# X = np.load("features.npy")
# y = np.load("labels.npy")

# classes = ["goldfish", "tiger_cat", "Egyptian_cat", "tiger", "elephant"]

# captions = [
#     "<start> a photo of a " + classes[i] + " <end>"
#     for i in y
# ]

# tok = Tokenizer()
# tok.fit_on_texts(captions)

# seq = tok.texts_to_sequences(captions)
# m = max(map(len, seq))
# seq = pad_sequences(seq, maxlen=m)

# inp = seq[:,:-1]
# out = seq[:,1:]

# vocab = len(tok.word_index) + 1
# n = m - 1

# img = Input((X.shape[1],))
# txt = Input((n,))

# a = RepeatVector(n)(Dense(128, activation="relu")(img))
# b = LSTM(128, return_sequences=True)(Embedding(vocab,64)(txt))

# z = Concatenate()([a,b])
# z = TimeDistributed(Dense(vocab, activation="softmax"))(z)

# model = Model([img,txt],z)
# model.compile(optimizer="adam",
#               loss="sparse_categorical_crossentropy")

# model.fit([X,inp],out,epochs=50,verbose=0)

# i = 0
# s = [tok.word_index["start"]]

# for _ in range(n):
#     p = pad_sequences([s], maxlen=n)
#     pred = model.predict([X[i:i+1],p],verbose=0)[0,len(s)-1]
#     word = np.argmax(pred)

#     if word == tok.word_index.get("end"):
#         break

#     s.append(word)

# print("Expected:", captions[i])
# print("Generated:", " ".join(tok.index_word[w] for w in s[1:]))