from pathlib import Path
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import TensorDataset

# We load the entire dataset into the GPU so that per-batch
# copies don't happen and slow down our code

__all__ = ["train_loader", "test_loader", "image_transform"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dsetname = "CIFAR100"
dsetnclasses = 100

dsetbase = Path("local_datasets/cifar100")

if not Path(dsetbase / "processed.pt").exists():
    # TODO: Use the actual mean and std
    transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root=dsetbase, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR100(
        root=dsetbase, train=False, download=True, transform=transform
    )

    train_data = torch.stack([trainset[i][0] for i in range(len(trainset))])
    train_labels = torch.tensor([trainset[i][1] for i in range(len(trainset))])

    test_data = torch.stack([testset[i][0] for i in range(len(testset))])
    test_labels = torch.tensor([testset[i][1] for i in range(len(testset))])

    torch.save((train_data, train_labels, test_data, test_labels), dsetbase / "processed.pt")


train_data, train_labels, test_data, test_labels = torch.load(dsetbase / "processed.pt")

train_dataset_gpu = TensorDataset(train_data.to(device), train_labels.to(device))
test_dataset_gpu = TensorDataset(test_data.to(device), test_labels.to(device))

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset_gpu, batch_size=32, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset_gpu, batch_size=128, shuffle=False, drop_last=True)

image_transform = v2.Compose(
    [
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
    ]
)
