import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

dsetname = "mini-imagenet"
dsetnclasses = 100

ds = load_dataset("timm/mini-imagenet")
ds.set_format(type="torch", columns=["image", "label"])

transform = v2.Compose(
    [
        v2.CenterCrop(224),
        v2.RGB(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def collate_fn(batch):
    images = torch.stack([transform(item["image"]) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return images, labels


train_loader = DataLoader(ds["train"], batch_size=256, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(ds["validation"], batch_size=512, shuffle=True, collate_fn=collate_fn)

image_transform = lambda x: x

# image_transform = v2.Compose(
#     [
#         # transforms.RandomCrop(224, padding=32),
#         # transforms.RandomHorizontalFlip(),
#     ]
# )
