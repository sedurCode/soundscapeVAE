import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
import wandb


class ConvolutionalVariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super(ConvolutionalVariationalAutoEncoder, self).__init__()
        encoder_modules = []
        decoder_modules = []
        # first layer is implicit
        encoder_modules.append(nn.Conv2d(in_channels=1,
                                         out_channels=config.output_sizes[0],
                                         kernel_size=config.kernels[0],
                                         stride=config.strides[0],
                                         padding=config.padding[0]))
        encoder_modules.append(nn.ReLU())

        for index in range(1, config.num_layers):
            encoder_modules.append(nn.Conv2d(in_channels=config.output_sizes[index - 1],
                                             out_channels=config.output_sizes[index],
                                             kernel_size=config.kernels[index],
                                             stride=config.strides[index],
                                             padding=config.padding[index]))
            if index != config.num_layers:
                encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)
        for index in range(config.num_layers-1, 0, -1):
            decoder_modules.append(nn.ConvTranspose2d(in_channels=config.output_sizes[index],
                                                      out_channels=config.output_sizes[index - 1],
                                                      kernel_size=config.kernels[index],
                                                      stride=config.strides[index],
                                                      padding=config.padding[index],
                                                      output_padding=config.output_padding[index]))
            decoder_modules.append(nn.ReLU())
        decoder_modules.append(nn.ConvTranspose2d(in_channels=config.output_sizes[0],
                                                  out_channels=1,
                                                  kernel_size=config.kernels[0],
                                                  stride=config.strides[0],
                                                  padding=config.padding[0],
                                                  output_padding=config.output_padding[0]))
        decoder_modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_modules)
        bottleneck_width = config.bottleneck_width
        self.fc1 = nn.Linear(config.output_sizes[-1], bottleneck_width)
        self.fc_mu = nn.Linear(bottleneck_width, bottleneck_width)
        self.fc_sigma = nn.Linear(bottleneck_width, bottleneck_width)
        self.fc2 = nn.Linear(bottleneck_width, config.output_sizes[-1])

    def forward(self, x):
        y = self.encoder(x)
        latent_dims = y.size()
        batch, _, _, _ = y.shape
        y = F.adaptive_avg_pool2d(y, 1).reshape(batch, -1) # y = torch.flatten(y, 1)
        hidden = self.fc1(y)
        mu = self.fc_mu(hidden)
        sigma = self.fc_sigma(hidden)
        z = self.reparameterize(mu, sigma)
        z = self.fc2(z)
        z = z.unsqueeze(2)  # z = z.unflatten(-1, sizes=(x.size(0), 64, 1, 1))
        z = z.unsqueeze(3)
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

    def construct_from_latent(self, latent_representations):
        z = self.fc2(latent_representations)
        z = z.unsqueeze(2)  # z = z.unflatten(-1, sizes=(x.size(0), 64, 1, 1))
        z = z.unsqueeze(3)
        reconstructed_predictions = self.decoder(z)
        return reconstructed_predictions

    def construct_from_distributions(self, mu, sigma):
        z = self.reparameterize(mu, sigma)
        z = self.fc2(z)
        z = z.unsqueeze(2)  # z = z.unflatten(-1, sizes=(x.size(0), 64, 1, 1))
        z = z.unsqueeze(3)
        z = self.decoder(z)
        return z


def final_loss(bce_loss, mu, sigma):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return BCE + KLD


if __name__ == "__main__":
    config = wandb.config
    config.num_layers = 4
    config.output_sizes = [8, 16, 32, 64]
    config.kernels = (4, 4, 4, 4)
    config.strides = (2, 2, 2, 2)
    config.padding = (1, 1, 1, 0)
    config.input_size = (1, 32, 32)
    config.bottleneck_width = 128
    config.output_padding = (0, 0, 0, 0)
    model = ConvolutionalVariationalAutoEncoder(config)
    summary(model.cuda(), (1, 32, 32))  # (colors, mel_bins, time)