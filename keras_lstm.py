import tensorflow as tf
import time
import numpy as np

def build_model():
    model = tf.keras.Sequential([
        # If stateful is True, it needs to know the batch size of the input as well
        tf.keras.layers.CuDNNGRU(64, return_sequences=True,
                                 recurrent_initializer='glorot_uniform',
                                 stateful=False),
    ])
    return model

# Training step
EPOCHS = 1

input = tf.keras.layers.Input(shape=[10, 3])
lstm = build_model()
outputs = lstm(input)
model = tf.keras.Model(inputs=input, outputs=outputs)

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    print(hidden)

    with tf.GradientTape() as tape:
        inp = np.random.rand(10, 10, 3).astype(np.float32)
        # feeding the hidden state back into the model
        # This is the interesting step
        predictions = model(inp)
        print(predictions.shape)

