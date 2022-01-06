import os
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import pandas as pd
import numpy as np


def main():
    print("TensorFlow version:", tf.__version__)
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)

    assets_subdir = 'wine-quality'
    filename = 'winequality-white.csv'
    sep = ';'
    root_dir = os.getcwd()
    file_abs_path = os.path.join(root_dir, '../datasplitting/assets/', assets_subdir, filename)
    header = 0
    col_names = None
    dtype1 = np.float32
    winedata = pd.read_csv(file_abs_path, sep=sep, header=header, names=col_names, dtype=dtype1)
    wine_features = winedata.copy()
    wine_labels = wine_features.pop('quality')
    wine_features = np.array(wine_features)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()  # returns Tuple of Numpy arrays
    print(wine_features)

    # x_train, x_test = x_train / 255.0, x_test / 255.0  # Convert the sample data from integers to floating-point numbers:

    normalize = tf.keras.layers.experimental.preprocessing.Normalization() # tf.keras.layers.Normalization()  #  # normalization layer
    normalize.adapt(wine_features)

    model = tf.keras.models.Sequential([
        normalize,
        tf.keras.layers.Dense(11, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    predictions = model(wine_features[:1]).numpy()
    print(predictions)

    error_fn = tf.keras.losses.MeanSquaredError()  # SparseCategoricalCrossentropy(from_logits=True)
    adaptive_lr = tf.optimizers.Adam()

    print(error_fn(wine_labels[:1], predictions).numpy())

    model.compile(optimizer=adaptive_lr,
                  loss=error_fn,
                  metrics=['mse'])

    model.fit(wine_features, wine_labels, epochs=50)
    # model.evaluate(x_test,  y_test, verbose=2)


if __name__ == '__main__':
    main()

