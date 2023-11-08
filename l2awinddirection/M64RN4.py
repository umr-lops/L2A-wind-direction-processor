import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, ZeroPadding2D, MaxPooling2D, Flatten, \
    Dense, Add, Lambda
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import math, keras

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pytz  # For timezone formating

from datetime import datetime

import l2awinddirection.utils

conf = l2awinddirection.utils.get_conf()

class M64RN4:
    def __init__(self, input_shape, data_augmentation):
        self.input_shape = input_shape
        self.data_augmentation = data_augmentation
        self.callback_path = None

    def load(self, path):
        self.model = keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def create(self):
        pass

    def custom_loss(self):
        pass

    def load_dataset(self, X_path, y_path):

        self.X_raw = np.load(X_path)
        self.y_raw = np.load(y_path)

        self.X_standardized = np.array([(x - np.average(x)) / np.std(x) for x in self.X_raw])

    def get_dataset(self, X_path, y_path):
        pass

    def create_and_compile(self, learning_rate, multi_gpus=False, verbose=False):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        nb_gpus = len(gpus)

        if verbose:
            print("Num GPUs Available: ", nb_gpus)

        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Divide the training between GPUS if there is more than one
        if (nb_gpus > 1) & multi_gpus:
            strategy = tf.distribute.MirroredStrategy()

            with strategy.scope():
                self.create()
                self.model.compile(optimizer, loss=self.custom_loss())

        else:
            self.create()
            self.model.compile(optimizer, loss=self.custom_loss())

    def launch_training(self, nb_epochs):
        callbacks = self.get_callbacks()
        train_history = self.model.fit(self.X_train, self.y_train,
                                       validation_data=(self.X_test, self.y_test),
                                       epochs=nb_epochs,
                                       shuffle=True,
                                       verbose=1,
                                       callbacks=callbacks)

    def infer(self, patch, norm=False):

        if norm:
            patch = (patch - np.average(patch)) / np.std(patch)

        pred = self.model.predict(patch.reshape(((1,) + self.input_shape)))
        return pred

    def get_callbacks(self):

        date = datetime.now(tz=pytz.timezone("Europe/Paris"))
        filepath = self.callback_path

        tail = '%s/%s/%s' % (self.model.name, datetime.strftime(date, "%d-%m-%y"), datetime.strftime(date, "%Hh%S"))

        if self.data_augmentation:
            filepath = "%s/w_data_augmentation/%s" % (filepath, tail)
        else:
            filepath = filepath = "%s/wo_data_augmentation/%s" % (filepath, tail)

        checkpoint_path = "%s/checkpoints/%s_epoch_{epoch:02d}__loss{val_loss:.2f}.hdf5" % (filepath, tail)
        log_path = "%s/logs" % (filepath)

        # Set up callbacks
        # Checkpoint to save the best model
        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')

        # Tensorboard callback to keep track of the training history
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

        # Stop the model if no improvement was made after 15 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=15)

        return checkpoint, tensorboard_callback, early_stopping

    def RNB_identity(self, X, f=3, channels=64):

        # Save the input value
        X_shortcut = X

        # First Layer
        X = ZeroPadding2D((1, 1))(X)
        X = Conv2D(filters=channels, kernel_size=(f, f), strides=(1, 1), padding="valid")(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        # Second Layer
        X = ZeroPadding2D((1, 1))(X)
        X = Conv2D(filters=channels, kernel_size=(f, f), strides=(1, 1), padding="valid")(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        # Add shortcut value to F(X), and pass it through RELU activation
        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)

        # Max Pooling and Batch Normalization
        X = MaxPooling2D((2, 2), strides=(2, 2))(X)
        X = BatchNormalization()(X)

        return X

    def RNB_convolutionnal(self, X, f=3, channels=64):

        # Save the input value
        X_shortcut = X

        # First Layer
        X = ZeroPadding2D((1, 1))(X)
        X = Conv2D(filters=channels, kernel_size=(f, f), strides=(1, 1), padding="valid")(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        # Second Layer
        X = ZeroPadding2D((1, 1))(X)
        X = Conv2D(filters=channels, kernel_size=(f, f), strides=(1, 1), padding="valid")(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        # Shortcut path
        X_shortcut = ZeroPadding2D((1, 1))(X_shortcut)
        X_shortcut = Conv2D(filters=channels, kernel_size=(f, f), strides=(1, 1), padding="valid")(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        # Add shortcut value to F(X), and pass it through RELU activation
        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)

        # Max Pooling and Batch Normalization
        X = MaxPooling2D((2, 2), strides=(2, 2))(X)
        X = BatchNormalization()(X)

        return X

    def M64RN4_body(self, X):

        if self.data_augmentation:
            X = Lambda(self.data_augmentation_layer, name="data_augmentation_layer")(X)

        # ResNet blocks
        X = self.RNB_convolutionnal(X)
        X = self.RNB_identity(X)
        X = self.RNB_identity(X)
        X = self.RNB_identity(X)

        # Fully connected network
        X = Flatten()(X)
        X = Dense((512))(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        X = Dense((128))(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        X = Dense((32))(X)
        X = Activation("relu")(X)
        X = BatchNormalization()(X)

        return X

    def data_augmentation_layer(self, x):
        return tf.image.rot90(x, np.random.choice([0, 2]))


class M64RN4_regression(M64RN4):

    def __init__(self, input_shape, data_augmentation):
        super().__init__(input_shape, data_augmentation)
        self.callback_path = conf['callback_path']

    def load_dataset(self, path_X_train, path_y_train, path_X_test, path_y_test):
        self.X_train = np.load(path_X_train)
        self.X_test = np.load(path_X_test)
        self.y_train = np.load(path_y_train)
        self.y_test = np.load(path_y_test)

        print('Shape X_train, X_test : %s, %s \nShape y_train, y_test : %s, %s' % (
        str(self.X_train.shape), str(self.X_test.shape), str(self.y_train.shape), str(self.y_test.shape)))

    def create(self):
        # Define the input shape
        X_input = Input(self.input_shape)
        X = self.M64RN4_body(X_input)

        # Output Layer
        X = Dense((1))(X)

        # Create model
        self.model = Model(inputs=X_input, outputs=X, name='M64RN4')

    def custom_loss(self):
        def regression_loss(y_true, y_pred):
            # Compute 1-cos(y-y_pred)**2
            loss = 1 - math.square(math.cos(math.subtract(y_pred, tf.reshape(y_true, [-1, 1]))))
            return loss

        return regression_loss

    def infer_test_set(self):
        self.y_pred = np.squeeze(self.model.predict(self.X_test, batch_size=64)) % np.pi

    def plot_results(self):
        # y_pred = [self.y_pred[i] for i in range(self.y_pred.shape[0])]
        y_pred = np.rad2deg(self.y_pred % np.pi)
        y_ref = np.rad2deg(self.y_test % np.pi)
        diff = (y_pred - y_ref) % 180

        plt.figure(figsize=(10, 8))
        plt.hist2d(y_ref, y_pred, 100, cmap='OrRd', cmin=1)
        plt.title('Predicted angle in function of the reference angle')
        plt.xlabel('Reference angle (°)')
        plt.ylabel('Angle predicted (°)')
        plt.colorbar(label='Number of values per bins')

        x = np.linspace(0, 360, 1000)
        plt.plot(x, x, color='k')
        plt.plot(x, x - 180, color='k')
        plt.plot(x, x + 180, color='k')

        plt.text(10, 165, 'Difference std = {:.2f}° \nDifference mean = {:.2f}°'.format(sp.stats.circstd(diff, 180),
                                                                                        sp.stats.circmean(diff, 180)),
                 backgroundcolor='white')
        plt.grid(linestyle='--', color='darkgray');


class M64RN4_distribution(M64RN4):

    def __init__(self, input_shape, data_augmentation, n_classes):
        super().__init__(input_shape, data_augmentation)
        # self.callback_path = '/raid/localscratch/?????/models_logs_and_save/distribution'
        self.callback_path = conf['callback_path']
        self.n_classes = n_classes

    def create(self):
        # Define the input shape
        X_input = Input(self.input_shape)
        X = self.M64RN4_body(X_input)

        # Output Layer
        X = Dense((self.n_classes))(X)
        X = Activation(tf.math.softmax)(X)

        # Create model
        self.model = Model(inputs=X_input, outputs=X, name='M64RN4')

    def custom_loss(self):
        return tf.keras.losses.CategoricalCrossentropy
