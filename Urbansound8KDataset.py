""""
Dataset stores all the information about a dataset
Dataloader is used to load and manage the data for training
The loader wraps the dataset
"""
import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd


class UrbanSound8KDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_directory,
                 transformation,
                 target_sample_rate,
                 target_num_samples,
                 device):
        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_directory
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
        self.min = 0
        self.max = 1

    def __len__(self):  # len(my_dataset_object)
        return len(self.annotations)

    def __getitem__(self, index):  # a_list[1] -> a_list.__getitem__(1)
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path,
                                     normalize=True)
        # Send signal to device
        signal = signal.to(self.device)
        # signal -> (channels, samples) -> (1, 16000)
        signal = self._resample_if_needed(signal, sr)
        # signal -> (channels, samples) -> (2, 16000)
        signal = self._mix_down_if_needed(signal)
        signal = self._cut_if_needed(signal)
        signal = self._right_pad_if_needed(signal)
        mel_spectrogram = self.transformation(signal)
        mel_spectrogram = self._normalise(mel_spectrogram)
        return mel_spectrogram, label

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        file = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold, file)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

    def _resample_if_needed(self, signal, sr):
        if sr == self.target_sample_rate:
            return signal
        resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        resampler.to(self.device)
        signal = resampler(signal)
        return signal

    def _mix_down_if_needed(self, signal):
        if signal.shape[0] == 1:
            return signal
        signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_needed(self, signal):
        # signal -> Tensor -> (channels, samples)
        if signal.shape[1] > self.target_num_samples:
            signal = signal[:, :self.target_num_samples]
        return signal

    def _right_pad_if_needed(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.target_num_samples:
            pad_size = self.target_num_samples - signal_length
            last_dim_padding = (0, pad_size)  # (pre_pad_num, post_pad_num) paired over dims
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def _denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


def get_encoder_layers():
    layers = []   # in -> (N, 1, 513, 47)
    layers.append({'layer_in':  1,
                   'layer_out': 16,
                   'kernel': 3,
                   'stride': 2,
                   'activation': True})
    layers.append({'layer_in': 16,
                   'layer_out': 32,
                   'kernel': 3,
                   'stride': 2,
                   'activation': True})
    layers.append({'layer_in':  32,
                   'layer_out': 64,
                   'kernel': 3,
                   'stride': 2,
                   'activation': True})
    layers.append({'layer_in':  64,
                   'layer_out': 128,
                   'kernel': 3,
                   'stride': 2,
                   'activation': True})
    layers.append({'layer_in': 128,
                   'layer_out': 256,
                   'kernel': 3,
                   'stride': 2,
                   'activation': True})
    layers.append({'layer_in': 256,
                   'layer_out': 512,
                   'kernel': 4,
                   'stride': 1,
                   'activation': False})
    return layers # out -> (N, 128, 21, 1)


def get_decoder_layers():
    layers = []
    layers.append({'layer_in': 512,
                   'layer_out': 256,
                   'kernel': 4,
                   'stride': 1,
                   'out_padding': 0,
                   'activation': True})
    layers.append({'layer_in': 256,
                   'layer_out': 128,
                   'kernel': 3,
                   'stride': 2,
                   'out_padding': 1,
                   'activation': True})
    layers.append({'layer_in': 128,
                   'layer_out': 64,
                   'kernel': 3,
                   'stride': 2,
                   'out_padding': 1,
                   'activation': True})
    layers.append({'layer_in': 64,
                   'layer_out': 32,
                   'kernel': 3,
                   'stride': 2,
                   'out_padding': 1,
                   'activation': True})
    layers.append({'layer_in': 32,
                   'layer_out': 16,
                   'kernel': 3,
                   'stride': 2,
                   'out_padding': 1,
                   'activation': True})
    layers.append({'layer_in': 16,
                   'layer_out': 1,
                   'kernel': 3,
                   'stride': 2,
                   'out_padding': 1,
                   'activation': False})
    return layers  # out -> (N, 1, 513, 47)


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/media/sedur/data/datasets/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/media/sedur/data/datasets/urbansound8k/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 32512  # 22050
    print(os.environ['PATH'])
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                           n_fft=1024,
                                                           hop_length=512,
                                                           n_mels=64
                                                           )

    usd = UrbanSound8KDataset(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[1]
    print(signal.size())