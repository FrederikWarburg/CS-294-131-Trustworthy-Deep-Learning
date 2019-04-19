from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import h5py

def get_mnist_dataloaders(image_size, batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_multi_mnist_dataloaders(path,image_size=64, batch_size=128):
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_data = MultiMNIST(path,  train = True, transform = all_transforms)
    test_data = MultiMNIST(path, train = False, transform = all_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def get_fashion_mnist_dataloaders(batch_size=128):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST('../fashion_data', train=False,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train',
                        batch_size=64):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = datasets.LSUN(db_path=path_to_data, classes=[dataset],
                              transform=transform)

    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)


from torch.utils.data import Dataset

class MultiMNIST(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path = 'MNIST_synthetic.h5', train = True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

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

        if False:
            self.X = list(f['val_dataset'])
            self.y = list(f['val_labels'])
            self.seg = list(f['val_segmentations'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index, seg=False):

        img = self.X[index]
        target = self.y[index]
        seg = self.seg[index]

        if self.transform:
            img = self.transform(img)

        return img, target, seg