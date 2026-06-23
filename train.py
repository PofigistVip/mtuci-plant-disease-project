import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import timm

from dataset import get_dataloaders
from config import *

def main():
    MODEL_NAME = sys.argv[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    num_classes = len(classes)

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_acc = 0.0



    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        model.train()

        train_loss = 0

        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_acc:.2f}%"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            Path("models").mkdir(exist_ok=True)

            torch.save(
                model.state_dict(),
                f"models/{MODEL_NAME}.pth"
            )

            print("Model saved")

    print(
        f"\nBest Validation Accuracy: {best_acc:.2f}%"
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()