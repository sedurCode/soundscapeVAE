import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchaudio
import torchvision
from Urbansound8KDataset import UrbanSound8KDataset
from FreeSpokenDigitDataset import FreeSpokenDigitDataset
from variationalautoencoder import VariationalAutoEncoder,  loss_function, get_encoder_layers, get_decoder_layers
import wandb
import numpy as np
from tqdm import tqdm


def train_single_epoch(model, loop, loss_fn, optimiser, device, epoch, reconstruction_rate, epochs):
    model.train()
    elem_str = f"Loss_{epoch}"
    for batch_idx, (input, _) in loop:
        optimiser.zero_grad()  # backpropagate error and update weights
        input = input.to(device)
        output, mu, logvar = model(input)  # calculate loss
        loss = loss_function(output, input, mu, logvar, reconstruction_rate)  # loss = loss_fn(prediction, input)  # loss = loss_fn(prediction, target)
        loss.backward()
        optimiser.step()
        loop.set_description(f"Epoch: [{epoch}/{epochs}]")
        loop.set_postfix(loss=loss.item())
    wandb.log({"Epoch_Loss": loss.item()})
    return loss.item()


def train(model, data_loader, loss_fn, optimiser, scheduler, device, epochs, config):
    for i in range(epochs):
        loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
        loss = train_single_epoch(model, loop, loss_fn, optimiser, device, i, config.reconstruction_rate, epochs)
        scheduler.step(loss)
    print("Finished training")


def test(model, device, test_loader):
    pass


if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 500
    LEARNING_RATE = 1e-5  # 1e-3
    WEIGHTING_DECAY = 1e-5
    # ANNOTATIONS_FILE = "/media/sedur/data/datasets/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
    # AUDIO_DIR = "/media/sedur/data/datasets/urbansound8k/UrbanSound8K/audio"
    AUDIO_DIR = "/media/sedur/data/datasets/fsdd/recordings"
    ANNOTATIONS_FILE = "/media/sedur/data/datasets/fsdd/annotations.csv"
    # SAMPLE_RATE = 22050
    # NUM_SAMPLES = 32512  # 22050
    SAMPLE_RATE = 24000
    NUM_SAMPLES = 32512  # 22050
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    n_workers = 20  # 4
    encoder_layers = get_encoder_layers()
    decoder_layers = get_decoder_layers()
    wandb.init(project="pytorch-soundscapevae")
    config = wandb.config
    config.dataset = "freespokendigitdataset"
    config.log_interval = 10 
    config.batch_size = BATCH_SIZE  # input batch size for training (default: 64)
    config.epochs = EPOCHS  # number of epochs to train (default: 10)
    config.lr = LEARNING_RATE  # learning rate (default: 0.01)
    config.lr_gamma = 0.9
    config.no_cuda = False  # disables CUDA training
    config.seed = 42  # random seed (default: 42)
    config.encoder_layers = encoder_layers
    config.decoder_layers = decoder_layers
    config.bottleneck_width = 7680
    config.reconstruction_rate = 10
    torch.manual_seed(config.seed)
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
    #                                                        n_fft=1024,
    #                                                        hop_length=512,
    #                                                        n_mels=64,
    #                                                        norm='slaney'
    #                                                        )
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1023,
                                                    hop_length=512,
                                                    normalized=True
                                                    )
    # usd = UrbanSound8KDataset(ANNOTATIONS_FILE,
    #                           AUDIO_DIR,
    #                           mel_spectrogram,
    #                           SAMPLE_RATE,
    #                           NUM_SAMPLES,
    #                           torch.device("cpu"))
    fsdd = FreeSpokenDigitDataset(ANNOTATIONS_FILE,
                                  AUDIO_DIR,
                                  spectrogram,
                                  SAMPLE_RATE,
                                  NUM_SAMPLES,
                                  torch.device("cpu"))
    dataloader = DataLoader(fsdd,
                            batch_size=BATCH_SIZE,
                            num_workers=n_workers,
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True)
    model = VariationalAutoEncoder(config)  # model = Net()
    model.to(device)
    loss_fn = nn.MSELoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=LEARNING_RATE,
    #                              weight_decay=WEIGHTING_DECAY
    #                              )
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE
                                 )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    wandb.watch(model, log="all", log_freq=100)
    train(model, dataloader, loss_fn, optimizer, scheduler, device, EPOCHS, config)
    torch.save(model.state_dict(), "fsdd_model.h5")
    wandb.save('fsdd_model.h5')
    wandb.finish()
    print(f"Finished")
