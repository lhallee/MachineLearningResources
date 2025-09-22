from torchvision import datasets, transforms


def get_mnist_dataset():
    train_dataset = datasets.MNIST('../data/MNIST/', download=True, train=True)
    mean = train_dataset.data.float().mean() / 255
    std = train_dataset.data.float().std() / 255
    print(f'Mean: {mean}, Std: {std}')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_dataset = datasets.MNIST('../data/MNIST/', download=True, train=True, transform=transform)
    test_dataset = datasets.MNIST('../data/MNIST/', download=True, train=False, transform=transform)
    return train_dataset, test_dataset
