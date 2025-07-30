
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import numpy as np

from PIL import Image
import os
import requests
import zipfile

class TinyImageNet(Dataset):
    """
    Custom Dataset class for Tiny ImageNet.
    Handles downloading, parsing, and loading the data.
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    dataset_folder = 'tiny-imagenet-200'

    def __init__(self, root, split='train', transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.data_path = os.path.join(self.root, self.dataset_folder)

        if download:
            self.download()

        if not os.path.isdir(self.data_path):
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self.class_to_idx, self.idx_to_class, self.class_names = self._get_class_info()
        self.image_paths, self.labels = self._load_data()

    def _get_class_info(self):
        wnids_path = os.path.join(self.data_path, 'wnids.txt')
        words_path = os.path.join(self.data_path, 'words.txt')

        with open(wnids_path, 'r') as f:
            wnids = [x.strip() for x in f]
        
        wnids.sort()
        class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        idx_to_class = {i: wnid for i, wnid in enumerate(wnids)}

        class_names = {}
        with open(words_path, 'r') as f:
            for line in f:
                wnid, name = line.strip().split('\t', 1)
                if wnid in class_to_idx:
                    class_names[wnid] = name
        
        # Sort class names by index
        sorted_class_names = [class_names[idx_to_class[i]] for i in range(len(wnids))]

        return class_to_idx, idx_to_class, sorted_class_names

    def _load_data(self):
        image_paths = []
        labels = []

        if self.split == 'train':
            train_path = os.path.join(self.data_path, 'train')
            for class_folder in os.listdir(train_path):
                class_path = os.path.join(train_path, class_folder, 'images')
                if os.path.isdir(class_path):
                    class_idx = self.class_to_idx[class_folder]
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        image_paths.append(img_path)
                        labels.append(class_idx)
        elif self.split == 'val':
            val_path = os.path.join(self.data_path, 'val')
            annotations_path = os.path.join(val_path, 'val_annotations.txt')
            val_images_path = os.path.join(val_path, 'images')
            
            with open(annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_id = parts[0], parts[1]
                    img_path = os.path.join(val_images_path, img_name)
                    image_paths.append(img_path)
                    labels.append(self.class_to_idx[class_id])
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train' or 'val'.")

        return image_paths, labels

    def download(self):
        zip_path = os.path.join(self.root, self.filename)
        if os.path.exists(self.data_path):
            print('Files already downloaded and verified.')
            return

        # Download the zip file
        print(f"Downloading {self.url} to {zip_path}")
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file
        print(f"Extracting {zip_path} to {self.root}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        # Clean up the zip file
        os.remove(zip_path)
        print("Download and extraction complete.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a PyTorch tensor.
    
    This class can produce repeatable results by providing a seed to its
    internal random number generator.
    """
    def __init__(self, mean: float = 0., std: float = 0.0, seed=42):
        """
        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
            seed (int, optional): A random seed for the generator. If provided,
                                  the noise added will be the same every time.
                                  Defaults to None.
        """
        self.std = std
        self.mean = mean
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the noise transformation.

        Args:
            tensor (torch.Tensor): The input tensor to which noise will be added.

        Returns:
            torch.Tensor: The tensor with added Gaussian noise.
        """
        # Use the internal generator if it exists, otherwise use PyTorch's global RNG
        noise = torch.randn(
            tensor.size(), 
            generator=self.generator
        ) * self.std + self.mean
        
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self) -> str:
        # Get the seed if the generator exists
        seed_info = f", seed={self.generator.initial_seed()}" if self.generator else ""
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}{seed_info})"
    
def get_subset(dataset, percentage,seed=42):
    rng = np.random.default_rng(seed)
    num_items = len(dataset)
    indices = list(range(num_items))
    split = int(np.floor(percentage * num_items))
    # np.random.shuffle(indices)
    # rng.shuffle(indices)
    subset_indices = indices[:split]
    return Subset(dataset, subset_indices)


class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def get_iris_dataloaders(batch_size, seed, generator):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    train_loader = DataLoader(IrisDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(IrisDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, {'input_size': X_train.shape[1], 'num_classes': len(iris.target_names), 'class_names': iris.target_names}

def get_image_dataloaders(dataset_name, batch_size, generator, transfer_learning, noise_level=0.0,daset_percentage=1.0,seed=42):

    if transfer_learning:
        # transform = transforms.Compose([
        #     transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
        

        if noise_level > 0:
            transform_list.append(AddGaussianNoise(mean=0., std=noise_level, seed=seed))
            # Normalization should be the last step
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        transform = transforms.Compose(transform_list)
        
    else:
        if dataset_name == 'tiny_imagenet':
            transform_list = [transforms.Resize((64, 64)), transforms.ToTensor()]
        else:
            transform_list = []
        if dataset_name == 'mnist':
            transform_list.append(transforms.Grayscale(num_output_channels=3))
            
        transform_list.append(transforms.ToTensor())
        
        if noise_level > 0:
            transform_list.append(AddGaussianNoise(mean=0., std=noise_level))
        
        transform = transforms.Compose(transform_list)


    if dataset_name == 'mnist':
        train_dset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        num_classes = 10
        class_names = [str(i) for i in range(num_classes)]
    elif dataset_name == 'cifar10':
        train_dset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        num_classes = 10
        class_names = [str(i) for i in range(num_classes)]
    elif dataset_name == 'cifar100':
        train_dset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dset = datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
        num_classes = 100
        class_names = [str(i) for i in range(num_classes)]
    elif dataset_name == 'stl10':
        train_dset = datasets.STL10(root='./data', split='train', transform=transform, download=True)
        test_dset = datasets.STL10(root='./data', split='test', transform=transform, download=True)
        num_classes = 10
        class_names = [str(i) for i in range(num_classes)]

    elif dataset_name == 'tiny_imagenet':
        # Use the custom TinyImageNet class
        train_dset = TinyImageNet(root='./data', split='train', transform=transform, download=True)
        # The test set in TinyImageNet is called 'val'
        test_dset = TinyImageNet(root='./data', split='val', transform=transform, download=True)
        num_classes = 200
        # Get class names from our custom dataset object
        class_names = train_dset.class_names
    else: raise ValueError(f"Unknown image dataset: {dataset_name}")
    
    # Apply dataset percentage
    if daset_percentage < 1.0:
        train_dset = get_subset(train_dset, daset_percentage, seed)
        test_dset = get_subset(test_dset, daset_percentage, seed)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, {'num_classes': num_classes, 'class_names': [str(i) for i in range(num_classes)]}

def get_data(dataset_name, batch_size, seed, generator, transfer_learning=False, noise_level=0.0, daset_percentage=1.0):
    if dataset_name == 'iris': 
        return get_iris_dataloaders(batch_size, seed, generator)
    elif dataset_name in ['mnist', 'cifar10','cifar100','stl10','tiny_imagenet']: 
        return get_image_dataloaders(dataset_name, batch_size, generator, transfer_learning, noise_level, daset_percentage, seed)
    else: raise ValueError(f"Dataset '{dataset_name}' not supported.")
