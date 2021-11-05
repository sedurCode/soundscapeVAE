import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
import wandb


class VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super(VariationalAutoEncoder, self).__init__()
        modules = []
        for layer in config.encoder_layers:
            modules.append(nn.Conv2d(in_channels=layer['layer_in'],
                                     out_channels=layer['layer_out'],
                                     kernel_size=layer['kernel'],
                                     stride=layer['stride'],
                                     padding=1))
            if layer['activation'] is True:
                modules.append(nn.ReLU())
        bottleneck_width = config.bottleneck_width
        self.encoder = nn.Sequential(*modules)
        self.fc21 = nn.Linear(bottleneck_width, bottleneck_width)
        self.fc22 = nn.Linear(bottleneck_width, bottleneck_width)
        self.fc3 = nn.Linear(bottleneck_width, bottleneck_width)
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
        y = self.encoder(x)
        latent_dims = y.size()
        y = torch.flatten(y, 1)
        mu = self.fc21(y)
        sigma = self.fc22(y)
        z = self.reparameterize(mu, sigma)
        z = self.fc3(z)
        z = z.unsqueeze(2) ## z = z.unflatten(-1, sizes=(x.size(0), 64, 1, 1))
        z = z.unsqueeze(3)
        z = torch.reshape(z, latent_dims)
        z = self.decoder(z)
        return z, mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps*std

    def reconstruct(self, spectrograms):
        reconstructed_predictions, mu, sigma = self.forward(spectrograms)
        latent_representations = self.reparameterize(mu, sigma)
        return reconstructed_predictions, latent_representations

#
# class VariationalAutoEncoder(nn.Module):
#     def __init__(self, config):
#         super(VariationalAutoEncoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1,  # N, 1,  64, 64
#                                out_channels=16,  # N, 16, 32, 32
#                                kernel_size=3,
#                                stride=2,
#                                padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16,  # N, 16, 32, 32
#                                out_channels=32,  # N, 32, 16, 16
#                                kernel_size=3,
#                                stride=2,
#                                padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32,  # N, 32, 16, 16
#                                out_channels=64,  # N, 64, 1, 1
#                                kernel_size=16)
#         self.fc21 = nn.Linear(64, 64)
#         self.fc22 = nn.Linear(64, 64)
#         self.fc23 = nn.Linear(64, 64)
#         self.deconv1 = nn.ConvTranspose2d(in_channels=64,  # N, 64, 1, 1
#                                           out_channels=32,  # N, 32, 16, 16
#                                           kernel_size=16)
#
#         self.deconv2 = nn.ConvTranspose2d(in_channels=32,  # N, 32, 16, 16
#                                           out_channels=16,  # N, 16, 32, 32
#                                           kernel_size=3,
#                                           stride=2,
#                                           padding=1,
#                                           output_padding=1)
#         self.deconv3 = nn.ConvTranspose2d(in_channels=16,  # N, 16, 32, 32
#                                           out_channels=1,  # N, 1, 64, 64
#                                           kernel_size=3,
#                                           stride=2,
#                                           padding=1,
#                                           output_padding=1)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # encoded = self.encoder(x)
#         x = F.relu(self.conv2(x))
#         x = self.conv3(x)
#         y = torch.flatten(x, 1)
#         mu = self.fc21(y)
#         sigma = self.fc22(y)
#         z = self.reparameterize(mu, sigma)
#         z = self.fc23(z)
#         z = z.unsqueeze(2) ## z = z.unflatten(-1, sizes=(x.size(0), 64, 1, 1))
#         z = z.unsqueeze(3)
#         z = F.relu(self.deconv1(z))  # decoded = self.decoder(encoded)
#         z = F.relu(self.deconv2(z))
#         z = torch.sigmoid(self.deconv3(z))
#         return z, mu, sigma
#
#     def reparameterize(self, mu, sigma):
#         std = torch.exp(0.5 * sigma)
#         eps = torch.randn_like(std)
#         return mu + eps*std
#
#     def reconstruct(self, spectrograms):
#         reconstructed_predictions, mu, sigma = self.forward(spectrograms)
#         latent_representations = self.reparameterize(mu, sigma)
#         return reconstructed_predictions, latent_representations


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(output, input, mu, sigma, reconstruction_loss_weight):
    BCE = F.binary_cross_entropy(output, input, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return reconstruction_loss_weight * BCE + KLD


# def get_encoder_layers():
#     layers = []
#     layers.append({'layer_in':  1,
#                    'layer_out': 16,
#                    'kernel': 3,
#                    'stride': 2,
#                    'activation': True})
#     layers.append({'layer_in': 16,
#                    'layer_out': 32,
#                    'kernel': 3,
#                    'stride': 2,
#                    'activation': True})
#     layers.append({'layer_in':  32,
#                    'layer_out': 64,
#                    'kernel': 16,
#                    'stride': 3,
#                    'activation': False})
#     return layers
#
#
# def get_decoder_layers():
#     layers = []
#     layers.append({'layer_in': 64,
#                    'layer_out': 32,
#                    'kernel': 16,
#                    'stride': 1,
#                    'out_padding': 0,
#                    'activation': True})
#     layers.append({'layer_in': 32,
#                    'layer_out': 16,
#                    'kernel': 3,
#                    'stride': 2,
#                    'out_padding': 1,
#                    'activation': True})
#     layers.append({'layer_in': 16,
#                    'layer_out': 1,
#                    'kernel': 3,
#                    'stride': 2,
#                    'out_padding': 1,
#                    'activation': False})
#     return layers

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
    encoder_layers = get_encoder_layers()
    decoder_layers = get_decoder_layers()
    config = wandb.config
    
    config.encoder_layers = encoder_layers
    config.decoder_layers = decoder_layers
    config.bottleneck_width = 7680
    
    model = VariationalAutoEncoder(config)
    summary(model.cuda(), (1, 512, 64))
    # summary(model.cuda(), (1, 64, 64))  # (colors, mel_bins, time)