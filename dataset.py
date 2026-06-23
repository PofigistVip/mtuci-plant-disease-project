from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import *

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def get_dataloaders():

    train_dataset = datasets.ImageFolder(
        f"{DATA_DIR}/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        f"{DATA_DIR}/val",
        transform=val_transform
    )

    test_dataset = datasets.ImageFolder(
        f"{DATA_DIR}/test",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )