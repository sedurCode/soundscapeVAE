import librosa
import torchaudio
import torch
import numpy as np
# from preprocess import MinMaxNormaliser


class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        # self._min_max_normaliser = MinMaxNormaliser(0, 1)
        self.min = 0
        self.max = 1

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        transform = torchaudio.transforms.GriffinLim(n_fft=1023)
        signals = []
        for spectrogram in spectrograms:
            # reshape the log spectrogram
            log_spectrogram = spectrogram.squeeze(0) # remove chans dim
            # apply denormalisation
            denorm_log_spec = self._denormalise(
                log_spectrogram, min_max_values["min"], min_max_values["max"])
            # log spectrogram -> spectrogram
            spec = torchaudio.functional.DB_to_amplitude(denorm_log_spec, 1.0, 0.5)
            # apply Griffin-Lim
            signal = transform(spec)
            # append signal to "signals"
            signals.append(signal)
        return signals

    def _normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def _denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array