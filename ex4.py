
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Concatenate
from tensorflow.keras.models import Model

# 1. Load dataset
path = "imagenet_subset"
train, test = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="both", seed=1,
    image_size=(224, 224), batch_size=4
)
classes = train.class_names

# 2. CNN encoder: extract image features
cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract(ds):
    X, Y = [], []
    for images, labels in ds:
        f = cnn.predict(preprocess_input(images.numpy()), verbose=0)
        X.extend(f)
        Y.extend(labels.numpy())
    return np.array(X), np.array(Y)

X_train, y_train = extract(train)
X_test, y_test = extract(test)

# 3. Create captions from class labels
captions = ["startseq a photo of " + c.replace("_", " ") + " endseq"
            for c in classes]

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(captions)
vocab = len(tokenizer.word_index) + 1
seqs = tokenizer.texts_to_sequences(captions)
maxlen = max(len(s) for s in seqs)

# 4. Prepare caption sequences for next-word prediction
texts = ["startseq a photo of " + classes[int(y)].replace("_", " ") + " endseq"
         for y in y_train]
seqs = tokenizer.texts_to_sequences(texts)

A, B = [], []
for f, s in zip(X_train, seqs):
    s = pad_sequences([s], maxlen=maxlen, padding="post")[0]
    A.append((f, s[:-1]))
    B.append(s[1:])

features = np.array([a[0] for a in A])
words = np.array([a[1] for a in A])
targets = tf.keras.utils.to_categorical(B, num_classes=vocab)

# 5. CNN-LSTM decoder
image_in = Input(shape=(2048,))
word_in = Input(shape=(maxlen - 1,))
img = RepeatVector(maxlen - 1)(image_in)
txt = Embedding(vocab, 32)(word_in)
x = Concatenate()([img, txt])
x = LSTM(64, return_sequences=True)(x)
out = Dense(vocab, activation="softmax")(x)

model = Model([image_in, word_in], out)
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit([features, words], targets, epochs=30, verbose=0)

# 6. Generate captions for test images
index = {v: k for k, v in tokenizer.word_index.items()}

for f, label in zip(X_test, y_test):
    seq = [tokenizer.word_index["startseq"]]

    for _ in range(maxlen - 1):
        padded = pad_sequences([seq], maxlen=maxlen - 1, padding="post")
        p = model.predict([f.reshape(1, -1), padded], verbose=0)
        next_id = int(np.argmax(p[0, len(seq) - 1]))

        if next_id == 0 or index.get(next_id) == "endseq":
            break
        seq.append(next_id)

    caption = " ".join(index.get(i, "") for i in seq[1:])
    print("Expected:", classes[int(label)])
    print("Generated:", caption)

# Architecture
# Image → CNN encoder → image feature vector → LSTM decoder → words → caption
# What is an RNN?
# A Recurrent Neural Network is designed for sequential data. It processes one step at a time and carries information from earlier steps forward. It can be used for text, time series and other sequences.
# What is an LSTM?
# LSTM means Long Short-Term Memory. It is a type of RNN designed to learn longer-range dependencies using a cell state and gates.

# Three gates:
# - Forget gate: decides what old information to retain or discard.
# - Input gate: decides what new information to store.
# - Output gate: controls what information is exposed as the hidden state.
# CNN encoder vs LSTM decoder
# - CNN encoder: extracts visual information from the image.
# - LSTM decoder: uses the image representation and preceding words to predict the next word.
# Tokenization and vocabulary
# - Tokenization: convert text into tokens, often words.
# - Vocabulary: the collection of distinct tokens.
# - Integer sequence: represent each token by an integer ID.
# - <start> token: signals the beginning of a caption.
# - <end> token: signals the end of a caption.
# - Padding: make sequences the same length for batch processing.