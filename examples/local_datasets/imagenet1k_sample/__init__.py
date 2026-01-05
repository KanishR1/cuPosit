from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

# We load the entire dataset into the GPU so that per-batch
# copies don't happen and slow down our code

__all__ = ["train_loader", "test_loader", "image_transform"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dsetname = "IMAGENET1K-Sample"
dsetnclasses = 1000

dsetbase = Path("local_datasets/imagenet1k_sample")

if not Path(dsetbase / "processed.pt").exists():
    # TODO: Use the actual mean and std
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    images = sorted([i for i in Path(dsetbase / "samples").iterdir() if i.suffix.lower() == ".jpeg"])

    train_data = torch.stack(
        [transform(torchvision.io.decode_image(image_path, mode="RGB")) for image_path in images[:900]]
    )
    train_labels = torch.tensor(range(len(images))[:900])

    test_data = torch.stack(
        [transform(torchvision.io.decode_image(image_path, mode="RGB")) for image_path in images[900:]]
    )
    test_labels = torch.tensor(range(len(images))[900:])

    torch.save((train_data, train_labels, test_data, test_labels), dsetbase / "processed.pt")


train_data, train_labels, test_data, test_labels = torch.load(dsetbase / "processed.pt")

train_dataset_gpu = TensorDataset(train_data.to(device), train_labels.to(device))
test_dataset_gpu = TensorDataset(test_data.to(device), test_labels.to(device))

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset_gpu, batch_size=128, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset_gpu, batch_size=100, shuffle=False, drop_last=True)

image_transform = transforms.Compose(
    [
        transforms.RandomCrop(224, padding=32),
        transforms.RandomHorizontalFlip(),
    ]
)
