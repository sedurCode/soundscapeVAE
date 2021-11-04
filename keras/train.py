import tensorflow as tf
import numpy as np
import wandb
import os
from wandb.keras import WandbCallback
from variationalautoencoder import VAE

SPECTROGRAMS_PATH = "/media/sedur/data/datasets/fsdd/spectrograms/"

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames) <- gotta add 1 more dim for 3 dims ex. RGB chans
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (samples, n_bins, n_frames, 1)
    return x_train


def train(x_train, config):
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    vae.summary()
    vae.compile(config)
    vae.train(x_train, config)
    return vae


if __name__ == "__main__":
    run = wandb.init(project="soundscapeVAE",
                     config={"learning_rate": 0.0001,
                             "batch_size": 64,
                             "epochs": 400,
                             "validation_split": 0.1,
                             "loss_function": "MeanSquaredError",
                             "optimizer": 'Adam',
                             "shuffle": True,
                             "architecture": "ConvAutoEncoder",
                             "dataset": "audio_mnist"})

    config = wandb.config
    x_train = load_fsdd(SPECTROGRAMS_PATH)  # x_train, y_train, x_test, y_test = load_mnist()
    vae = train(x_train, config)
    vae.save("model")
    run.finish()
    vae2 = VAE.load("model")
    vae2.summary()
