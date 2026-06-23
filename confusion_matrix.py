import sys

import torch
import timm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from dataset import get_dataloaders

def main():
    MODEL_NAME = sys.argv[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, test_loader, classes = get_dataloaders()

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(classes)
    )

    model.load_state_dict(
        torch.load(
            f"models/{MODEL_NAME}.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(14, 14))

    sns.heatmap(
        cm,
        cmap="Blues"
    )

    plt.title(
        f"Confusion Matrix - {MODEL_NAME}"
    )

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()