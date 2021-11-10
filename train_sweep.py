import os.path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchaudio
import torchvision
from matplotlib import pyplot as plt
from Urbansound8KDataset import UrbanSound8KDataset
from FreeSpokenDigitDataset import FreeSpokenDigitDataset
from ConvolutionalVariationalAutoEncoder import ConvolutionalVariationalAutoEncoder, final_loss
import imageio
import wandb
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib
from torchvision.utils import make_grid
matplotlib.style.use('ggplot')
to_pil_image = torchvision.transforms.ToPILImage()


def image_to_vid(images, outputs_dir):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(os.path.join(outputs_dir, 'generated_images.gif', imgs))


def save_reconstructed_images(recon_images, epoch, outputs_dir):
    save_image(recon_images.cpu(), os.path.join(outputs_dir, f"output{epoch}.jpg"))


def save_loss_plot(train_loss, valid_loss, outputs_dir):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, 'loss.jpg'))
    plt.show()


def plot_latent(model, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = model.reconstruct(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


def train_single_epoch(model, loop, loss_fn, optimiser, device, epoch, epochs):
    model.train()
    for batch_idx, (input, _) in loop:
        optimiser.zero_grad()  # backpropagate error and update weights
        input = input.to(device)
        reconstruction, mu, logvar = model(input)  # calculate loss
        bce_loss = loss_fn(reconstruction, input)
        loss = final_loss(bce_loss, mu, logvar)  # loss = loss_fn(prediction, input)  # loss = loss_fn(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        loop.set_description(f"Epoch: [{epoch}/{epochs}]")
        loop.set_postfix(loss=loss.item())
    return loss.item()


def train(model, train_loader, test_loader, loss_fn, optimiser, scheduler, device, epochs, config, test_set, outputs_dir):
    grid_images = []
    for i in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loss = train_single_epoch(model, loop, loss_fn, optimiser, device, i, epochs)
        valid_epoch_loss, recon_images = validate(model, test_loader, test_set, device, loss_fn)
        wandb.log({"Train_Epoch_Loss": loss,
                   "Valid_epoch_loss": valid_epoch_loss})
        scheduler.step(loss)
        # save the reconstructed images from the validation loop
        save_reconstructed_images(recon_images, i + 1, outputs_dir)
        # convert the reconstructed images to PyTorch image grid format
        image_grid = make_grid(recon_images.detach().cpu())
        grid_images.append(image_grid)
    print("Finished training")
    return grid_images


def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            data = data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    n_workers = 20
    wandb.init(project="mnisting", entity="sedurcode")
    dataset_path = "/media/sedur/data/datasets/mnist/"
    outputs_path = os.path.join(dataset_path, "outputs/")
    config = wandb.config
    config.dataset = "mnist"
    config.log_interval = 10
    config.num_layers = 4
    config.output_sizes = [8, 16, 32, 64]
    config.kernels = (4, 4, 4, 4)
    config.strides = (2, 2, 2, 2)
    config.padding = (1, 1, 1, 0)
    config.input_size = (1, 32, 32)
    config.output_padding = (0, 0, 0, 0)
    config.bottleneck_width = 32
    config.reconstruction_rate = 10
    num_workers = 20
    torch.manual_seed(config.seed)
    model_name = wandb.run.name + "mnist_model.h5"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    outputs_dir = os.path.join('/media/sedur/data/datasets/mnist/outputs', wandb.run.name)
    if os.path.isdir(outputs_dir) is False:
        os.mkdir(outputs_dir)
    train_data = torchvision.datasets.MNIST(dataset_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    pull_batch_size = train_data.__len__()
    temp_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=pull_batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)
    x_train, y_train = next(iter(temp_loader))
    train_data.data.to(device)
    train_data.targets.to(device)
    train_tensor_data = torch.utils.data.TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = torch.utils.data.DataLoader(train_tensor_data,
                                               batch_size=config.batch_size,
                                               shuffle=True)
    test_data = torchvision.datasets.MNIST(root=dataset_path,
                                          train=False,
                                          download=True,
                                          transform=transform
                                          )
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config.batch_size,
                                              shuffle=True)
    model = ConvolutionalVariationalAutoEncoder(config).to(device) # model = Net()
    loss_fn = nn.BCELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = None
    wandb.watch(model, log="all", log_freq=100)
    grid_images = train(model, train_loader, test_loader, loss_fn, optimizer,
                        scheduler, device, config.epochs, config, test_data, outputs_dir)
    image_to_vid(grid_images, outputs_dir)
    torch.save(model.state_dict(), os.path.join(outputs_path, model_name))
    wandb.save(os.path.join(outputs_path, model_name))
    wandb.finish()
    print(f"Finished")
