import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import h5py
from netdissect import retain_layers, dissect
from netdissect import ReverseNormalize


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = int(self.img_size[0] / 16), int(self.img_size[1] / 16)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


generator = torch.load("gen_mnist_model_epoch_200.pt", map_location='cpu')


def get_multi_mnist_dataloaders(batch_size=128):
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(100),
        transforms.ToTensor()
    ])

    train_data = MultiMNIST('MNIST_synthetic.h5', train=True, transform=all_transforms)
    test_data = MultiMNIST('MNIST_synthetic.h5', train=False, transform=all_transforms)

    return train_data, test_data


class MultiMNIST(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path='MNIST_synthetic.h5', train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(MultiMNIST, self).__init__()
        self.path = path
        self.transform = transform

        f = h5py.File(self.path, 'r')

        if train:
            self.X = list(f['train_dataset'])
            self.y = list(f['train_labels'])
            self.seg = list(f['train_segmentations'])

        else:
            self.X = list(f['test_dataset'])
            self.y = list(f['test_labels'])
            self.seg = list(f['test_segmentations'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):  # , seg=False):
        print(index)
        img = self.X[index]
        target = self.y[index]
        seg = self.seg[index]

        if self.transform:
            img = self.transform(img)

        return img, target, seg


generator.eval()

retain_layers(generator, ['features_to_image.0',
                          'features_to_image.1',
                          'features_to_image.2',
                          'features_to_image.3',
                          'features_to_image.4',
                          'features_to_image.5',
                          'features_to_image.6',
                          'features_to_image.7',
                          'features_to_image.8',
                          'features_to_image.9',
                          'features_to_image.10'])
print("here")
bds, _ = get_multi_mnist_dataloaders()
print(bds)
dissect('sample_data/', generator, bds,
        recover_image=ReverseNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        batch_size=100,
        examples_per_unit=10)
