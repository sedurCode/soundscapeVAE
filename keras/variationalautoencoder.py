import tensorflow as tf
import numpy as np
import wandb
import os
import pickle
from wandb.keras import WandbCallback

tf.compat.v1.disable_eager_execution()


def scheduler(epoch, lr):
    if epoch < 170:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


class VAE:
    """VAE represents a deep convolutional variational autoencoder architecture with mirrored encoder and decoder"""
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        self.alpha = 1000000  # reconstruction loss weight

        self.encoder = None
        self.decoder = None
        self.model = None
        self._shape_before_bottleneck = None
        self._model_input = None

        self._num_conv_layers = len(self.conv_filters)

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, config):
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss
                           )

    def train(self, x_train, config):
        self.model.fit(x_train,  # input
                       x_train,  # output (expected)
                       batch_size=config.batch_size,
                       epochs=config.epochs,
                       shuffle=config.shuffle,
                       callbacks=[WandbCallback(), tf.keras.callbacks.LearningRateScheduler(scheduler)])

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def save(self, save_folder="."):
        self._create_folder(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        weights_path = os.path.join(save_folder, "weights.h5")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        return self.alpha * reconstruction_loss + kl_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + self.log_sigma - tf.keras.backend.square(self.mu) -
                                              tf.keras.backend.exp(self.log_sigma), axis=1)
        return kl_loss

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = tf.keras.Model(encoder_input, bottleneck, name="encoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = tf.keras.Model(model_input, model_output, name="autoencoder")

    def _add_encoder_input(self):
        return tf.keras.Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Creates all convolutional blocks in encoder"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (dense layer) with gaussian sampling"""
        self._shape_before_bottleneck = tf.keras.backend.int_shape(x)[1:]  # [2, 7, 7, 32]
        x = tf.keras.layers.Flatten()(x)
        self.mu = tf.keras.layers.Dense(self.latent_space_dim, name="mu")(x)
        self.log_sigma = tf.keras.layers.Dense(self.latent_space_dim, name="log_sigma")(x)

        def sample_point_from_normal_distribution(args):
            """Applies our mv_normal_distro term z=mu_vex+Sigma*epsilon"""
            mu, log_sigma = args
            sigma = tf.keras.backend.exp(log_sigma * 0.5)
            epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(self.mu), mean=0., stddev=1.)
            z = mu + sigma * epsilon
            return z
        x = tf.keras.layers.Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu,
                                                                                                  self.log_sigma])
        return x

    def _add_conv_layer(self, layer_index, x):
        """Addas a convolutional block to a graph of layers, consisting of | conv2d > ReLU > Batchnorm"""
        layer_number = layer_index + 1
        conv_layer = tf.keras.layers.Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = tf.keras.layers.ReLU(name=f"encoder_ReLU_{layer_number}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"encoder_batchnorm_{layer_number}")(x)
        return x

    def _add_decoder_input(self):
        return tf.keras.Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [4, 4, 32] -> 1024
        dense_layer = tf.keras.layers.Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return tf.keras.layers.Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add convolutional transpose blocks"""
        # Loop through all conv layers in reverse order and stop at first layer
        for layer_index in reversed(range(1, self._num_conv_layers)): # [0, 1, 2] -> [2, 1]
            x = self._add_conv_transpose_layer(x, layer_index)
        return x

    def _add_decoder_output(self, conv_transpose_layers):
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters=1,  # [24, 24, 1] <- one channel output
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(conv_transpose_layers)
        output_layer = tf.keras.layers.Activation(activation="sigmoid", name="sigmoid")(x)
        return output_layer

    def _add_conv_transpose_layer(self, x, layer_index):
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_number}"
        )
        x = conv_transpose_layer(x)
        x = tf.keras.layers.ReLU(name=f"decoder_ReLU_{layer_number}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"decoder_batchnorm_{layer_number}")(x)
        return x

    def _create_folder(self, save_folder):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    def _save_parameters(self, save_folder):
        save_path = os.path.join(save_folder, "parameters.pkl")
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


if __name__ == "__main__":
    run = wandb.init(project="soundscapeVAE",
                     config={"learning_rate": 0.0001,
                             "batch_size": 64,
                             "epochs": 800,
                             "validation_split": 0.1,
                             "loss_function": "mse_loss",
                             "optimizer": 'Adam',
                             "shuffle": True,
                             "architecture": "ConvAutoEncoder",
                             "dataset": "audio_mnist"})

    config = wandb.config
    vae = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=32
    )
    vae.summary()
    vae.compile(config)
    run.finish()