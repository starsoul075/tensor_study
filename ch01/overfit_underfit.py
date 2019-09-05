import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model.compile(optimizer=keras.optimizers.Adam(),
                       loss=keras.losses.BinaryCrossentropy(),
                       metrics=[keras.metrics.Accuracy(), keras.metrics.BinaryCrossentropy()])
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
#
# smaller_model = keras.Sequential([
#     keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(4, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# smaller_model.compile(optimizer=keras.optimizers.Adam(),
#                       loss=keras.losses.BinaryCrossentropy(),
#                       metrics=[keras.metrics.Accuracy(), keras.metrics.BinaryCrossentropy()])
# smaller_history = smaller_model.fit(train_data,
#                                     train_labels,
#                                     epochs=20,
#                                     batch_size=512,
#                                     validation_data=(test_data, test_labels),
#                                     verbose=2)
#
# bigger_model = keras.models.Sequential([
#     keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# bigger_model.compile(optimizer=keras.optimizers.Adam(),
#                      loss=keras.losses.BinaryCrossentropy(),
#                      metrics=[keras.metrics.Accuracy(), keras.metrics.BinaryCrossentropy()])
# bigger_history = bigger_model.fit(train_data, train_labels,
#                                   epochs=20,
#                                   batch_size=512,
#                                   validation_data=(test_data, test_labels),
#                                   verbose=2)
l2_mode = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
l2_mode.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.Accuracy(),
                         keras.metrics.BinaryCrossentropy()])
l2_mode_history = l2_mode.fit(train_data, train_labels,
                              epochs=20,
                              batch_size=512,
                              validation_data=(test_data, test_labels),
                              verbose=2)

dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
dpt_model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.Accuracy(), keras.metrics.BinaryCrossentropy()])
dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()
        plt.xlim([0, max(history.epoch)])
