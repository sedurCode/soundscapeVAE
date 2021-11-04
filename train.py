import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchaudio
import torchvision
from Urbansound8KDataset import UrbanSound8KDataset
from autoencoder import AutoEncoder, get_decoder_layers, get_encoder_layers
import wandb
import numpy as np


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    model.train()
    elem_str = f"Loss_{epoch}"
    for input, _ in data_loader:
        optimiser.zero_grad()  # backpropagate error and update weights
        input = input.to(device)
        prediction = model(input)  # calculate loss
        loss = loss_fn(prediction, input)  # loss = loss_fn(prediction, target)
        loss.backward()
        optimiser.step()

        wandb.log({elem_str: loss.item()})
    print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    wandb.log({"Epoch_Loss": loss.item()})


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, i)
    print("Finished training")


def test(model, device, test_loader):
    pass


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,  # N, 1,  64, 64
                               out_channels=16,  # N, 16, 32, 32
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,  # N, 16, 32, 32
                               out_channels=32,  # N, 32, 16, 16
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,  # N, 32, 16, 16
                               out_channels=64,  # N, 64, 1, 1
                               kernel_size=16)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64,  # N, 64, 1, 1
                                          out_channels=32,  # N, 32, 16, 16
                                          kernel_size=16)

        self.deconv2 = nn.ConvTranspose2d(in_channels=32,  # N, 32, 16, 16
                                          out_channels=16,  # N, 16, 32, 32
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16,  # N, 16, 32, 32
                                          out_channels=1,  # N, 1, 64, 64
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # encoded = self.encoder(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(self.deconv1(x))  # decoded = self.decoder(encoded)
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

    def backward(self):
        pass


if __name__ == "__main__":
    BATCH_SIZE = 8
    EPOCHS = 200
    LEARNING_RATE = 1e-5  # 1e-3
    WEIGHTING_DECAY = 1e-5
    ANNOTATIONS_FILE = "/media/sedur/data/datasets/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/media/sedur/data/datasets/urbansound8k/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 32512  # 22050
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    n_workers = 20  # 4
    encoder_layers = get_encoder_layers()
    decoder_layers = get_decoder_layers()
    wandb.init(project="pytorch-audiovae")
    config = wandb.config
    config.log_interval = 10 
    config.batch_size = BATCH_SIZE  # input batch size for training (default: 64)
    config.epochs = EPOCHS  # number of epochs to train (default: 10)
    config.lr = LEARNING_RATE  # learning rate (default: 0.01)
    config.no_cuda = False  # disables CUDA training
    config.seed = 42  # random seed (default: 42)
    config.encoder_layers = encoder_layers
    config.decoder_layers = decoder_layers

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                           n_fft=1024,
                                                           hop_length=512,
                                                           n_mels=64,
                                                           norm='slaney'
                                                           )
    usd = UrbanSound8KDataset(ANNOTATIONS_FILE,
                              AUDIO_DIR,
                              mel_spectrogram,
                              SAMPLE_RATE,
                              NUM_SAMPLES,
                              torch.device("cpu"))
    dataloader = DataLoader(usd,
                            batch_size=BATCH_SIZE,
                            num_workers=n_workers,
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True)
    model = AutoEncoder(config)  # model = Net()
    model.to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHTING_DECAY
                                 )
    wandb.watch(model, log="all")
    train(model, dataloader, loss_fn, optimizer, device, EPOCHS)
    torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')
    wandb.finish()
    print(f"Finished")
