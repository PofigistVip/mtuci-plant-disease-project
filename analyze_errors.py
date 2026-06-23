import sys
from pathlib import Path

import torch
import timm

import matplotlib.pyplot as plt

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

    correct_dir = Path("results/correct")
    wrong_dir = Path("results/wrong")

    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    correct_count = 0
    wrong_count = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(1)

            for i in range(len(images)):

                image = images[i].cpu()

                true_class = classes[labels[i].item()]
                pred_class = classes[preds[i].item()]

                image = image.permute(1, 2, 0).numpy()

                image = image.clip(0, 1)

                if (
                    pred_class == true_class
                    and correct_count < 3
                ):

                    plt.figure(figsize=(5, 5))
                    plt.imshow(image)

                    plt.title(
                        f"TRUE: {true_class}\nPRED: {pred_class}"
                    )

                    plt.axis("off")

                    plt.savefig(
                        correct_dir /
                        f"correct_{correct_count+1}.png",
                        bbox_inches="tight"
                    )

                    plt.close()

                    correct_count += 1

                elif (
                    pred_class != true_class
                    and wrong_count < 3
                ):

                    plt.figure(figsize=(5, 5))
                    plt.imshow(image)

                    plt.title(
                        f"TRUE: {true_class}\nPRED: {pred_class}"
                    )

                    plt.axis("off")

                    plt.savefig(
                        wrong_dir /
                        f"wrong_{wrong_count+1}.png",
                        bbox_inches="tight"
                    )

                    plt.close()

                    wrong_count += 1

                if (
                    correct_count >= 3
                    and wrong_count >= 3
                ):
                    break

            if (
                correct_count >= 3
                and wrong_count >= 3
            ):
                break

    print(
        f"Saved {correct_count} correct "
        f"and {wrong_count} wrong examples."
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()