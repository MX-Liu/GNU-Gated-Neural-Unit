
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
import numpy as np

# 1. Define the custom transform class for adding noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)
    
    
def get_subset(dataset, percentage):
    num_items = len(dataset)
    indices = list(range(num_items))
    split = int(np.floor(percentage * num_items))
    np.random.shuffle(indices)
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

def get_image_dataloaders(dataset_name, batch_size, generator, transfer_learning, noise_level=0.0,daset_percentage=1.0):

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
            transform_list.append(AddGaussianNoise(mean=0., std=noise_level))
            # Normalization should be the last step
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        transform = transforms.Compose(transform_list)
        
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
    elif dataset_name == 'cifar10':
        train_dset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        num_classes = 10
    else: raise ValueError(f"Unknown image dataset: {dataset_name}")
    
    # Apply dataset percentage
    if daset_percentage < 1.0:
        train_dset = get_subset(train_dset, daset_percentage)
        test_dset = get_subset(test_dset, daset_percentage)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, {'num_classes': num_classes, 'class_names': [str(i) for i in range(num_classes)]}

def get_data(dataset_name, batch_size, seed, generator, transfer_learning=False, noise_level=0.0, daset_percentage=1.0):
    if dataset_name == 'iris': return get_iris_dataloaders(batch_size, seed, generator)
    elif dataset_name in ['mnist', 'cifar10']: return get_image_dataloaders(dataset_name, batch_size, generator, transfer_learning, noise_level, daset_percentage)
    else: raise ValueError(f"Dataset '{dataset_name}' not supported.")
