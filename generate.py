import os
import pickle
import numpy as np
import soundfile as sf
from soundgenerator import SoundGenerator
# from autoencoder import VAE
# from train import SPECTROGRAMS_PATH
import torch
import wandb
import torchaudio
from torch.utils.data import DataLoader, DistributedSampler
from variationalautoencoder import VariationalAutoEncoder, get_decoder_layers, get_encoder_layers
from FreeSpokenDigitDataset import FreeSpokenDigitDataset
HOP_LENGTH = 256
AUDIO_DIR = "/media/sedur/data/datasets/fsdd/recordings"
ANNOTATIONS_FILE = "/media/sedur/data/datasets/fsdd/annotations.csv"
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
# MIN_MAX_VALUES_PATH = "/home/valerio/datasets/fsdd/min_max_values.pkl"


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


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


if __name__ == "__main__":
    # initialise sound generator
    # vae = VAE.load("model")
    SAMPLE_RATE = 24000
    NUM_SAMPLES = 32256  # 22050
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    n_workers = 10  # 4
    encoder_layers = get_encoder_layers()
    decoder_layers = get_decoder_layers()
    config = wandb.config
    config.seed = 42  # random seed (default: 42)
    config.encoder_layers = encoder_layers
    config.decoder_layers = decoder_layers
    config.bottleneck_width = 7680
    torch.manual_seed(config.seed)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1023,
                                                    hop_length=512,
                                                    normalized=True
                                                    )
    vae = VariationalAutoEncoder(config)
    vae.load_state_dict(torch.load("fsdd_model.h5"))
    vae.eval()
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    # with open(MIN_MAX_VALUES_PATH, "rb") as f:
    #     min_max_values = pickle.load(f)

    # specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    fsdd = FreeSpokenDigitDataset(ANNOTATIONS_FILE,
                                  AUDIO_DIR,
                                  spectrogram,
                                  SAMPLE_RATE,
                                  NUM_SAMPLES,
                                  torch.device("cpu"))
    dataloader = DataLoader(fsdd,
                            batch_size=10,
                            num_workers=n_workers,
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True)
    grams, labels = next(iter(dataloader))
    print(f"{grams.size()}")
    # sample spectrograms + min max values
    # sampled_specs, sampled_min_max_values = select_spectrograms(specs,
    #                                                             file_paths,
    #                                                             min_max_values,
    #                                                             5)
    # sampled_specs = dataloader[1]

    # generate audio for sampled spectrograms
    sampled_min_max_values = {"min": -10,
                              "max": 10}
    signals, _ = sound_generator.generate(grams,
                                          sampled_min_max_values)
    print(f"{torch.max(signals[0])}")
    print(f"{torch.min(signals[0])}")

    signal = signals[0].detach().numpy()
    signal = signal / np.max(np.abs(signal))
    # convert spectrogram samples to audio
    # original_signals = sound_generator.convert_spectrograms_to_audio(
    #     sampled_specs, sampled_min_max_values)

    # save audio signals
    save_path = "gensig.wav"
    sf.write(save_path, signal, SAMPLE_RATE)
    # save_signals(signals, SAVE_DIR_GENERATED)
    # save_signals(original_signals, SAVE_DIR_ORIGINAL)






