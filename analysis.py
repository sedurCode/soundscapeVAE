import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from variationalautoencoder import VAE
import soundfile as sf
from soundgenerator import SoundGenerator
from variationalautoencoder import VAE
from train import SPECTROGRAMS_PATH

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
MIN_MAX_VALUES_PATH = "/media/sedur/data/datasets/fsdd/min_max_values.pkl"


def load_fsdd(spectrograms_path):
    x_train = []
    labels = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            labels.append(int(file_name[0]))
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, labels


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


# def load_fsdd(spectrograms_path):
#     x_train = []
#     file_paths = []
#     for root, _, file_names in os.walk(spectrograms_path):
#         for file_name in file_names:
#             file_path = os.path.join(root, file_name)
#             spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
#             x_train.append(spectrogram)
#             file_paths.append(file_path)
#     x_train = np.array(x_train)
#     x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
#     return x_train, file_paths


def select_images(images, labels, num_images=10):
    labels = np.array(labels)
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    vae = VAE.load("model")
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    specs, labels = load_fsdd(SPECTROGRAMS_PATH)
    num_images = 6000
    sample_images, sample_labels = select_images(specs, labels, num_images)
    _, latent_representations = vae.reconstruct(sample_images)
    # sample_labels = sample_labels / np.max(np.abs(sample_labels))  # sample_labels = sample_labels / np.max(np.abs(sample_labels))
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)

    # num_sample_images_to_show = 8
    # sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    # reconstructed_images, _ = vae.reconstruct(sample_images)
    # plot_reconstructed_images(sample_images, reconstructed_images)

    # num_images = 6000
    # sample_images, sample_labels = select_images(x_test, y_test, num_images)
    # _, latent_representations = vae.reconstruct(sample_images)
    # plot_images_encoded_in_latent_space(latent_representations, sample_labels)



# if __name__ == "__main__":
#     # initialise sound generator
#     vae = VAE.load("model")
#     sound_generator = SoundGenerator(vae, HOP_LENGTH)
#
#     # load spectrograms + min max values
#     with open(MIN_MAX_VALUES_PATH, "rb") as f:
#         min_max_values = pickle.load(f)
#
#     specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)
#
#     # sample spectrograms + min max values
#     sampled_specs, sampled_min_max_values = select_spectrograms(specs,
#                                                                 file_paths,
#                                                                 min_max_values,
#                                                                 5)
#
#     # generate audio for sampled spectrograms
#     signals, _ = sound_generator.generate(sampled_specs,
#                                           sampled_min_max_values, True)