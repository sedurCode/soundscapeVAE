from torch import nn
from torchsummary import summary
import wandb

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()
        modules = []
        for layer in config.encoder_layers:
            modules.append(nn.Conv2d(in_channels=layer['layer_in'],
                                     out_channels=layer['layer_out'],
                                     kernel_size=layer['kernel'],
                                     stride=layer['stride'],
                                     padding=1))
            if layer['activation'] is True:
                modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules)
        modules = []
        for layer in config.decoder_layers:
            modules.append(nn.ConvTranspose2d(in_channels=layer['layer_in'],
                                              out_channels=layer['layer_out'],
                                              kernel_size=layer['kernel'],
                                              stride=layer['stride'],
                                              padding=1,
                                              output_padding=layer['out_padding']))
            if layer['activation'] is True:
                modules.append(nn.ReLU())
        modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def predict(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def get_encoder_layers():
    layers = []
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
    layers.append({'layer_in':  128,
                   'layer_out': 256,
                   'kernel': 4,
                   'stride': 1,
                   'activation': False})
    return layers


def get_decoder_layers():
    layers = []
    layers.append({'layer_in': 256,
                   'layer_out': 128,
                   'kernel': 4,
                   'stride': 1,
                   'out_padding': 0,
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
    return layers


if __name__ == "__main__":
    encoder_layers = get_encoder_layers()
    decoder_layers = get_decoder_layers()
    config = wandb.config
    
    config.encoder_layers = encoder_layers
    config.decoder_layers = decoder_layers
    
    model = AutoEncoder(config)
    summary(model.cuda(), (1, 64, 64))  # (colors, mel_bins, time)