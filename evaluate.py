import sys

import torch
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import get_dataloaders

from pathlib import Path
import pandas as pd

import time

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

    dummy = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)

    # прогрев GPU
    for _ in range(20):
        _ = model(dummy)

    start = time.perf_counter()

    for _ in range(100):
        _ = model(dummy)

    end = time.perf_counter()

    inference_ms = (
        (end - start) / 100
    ) * 1000

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    model_size_mb = round(
        Path(
            f"models/{MODEL_NAME}.pth"
        ).stat().st_size / 1024 / 1024,
        2
    )

    result = {
        "Model": MODEL_NAME,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "Size_MB": model_size_mb,
        "Inference_ms": round(
            inference_ms,
            2
        )
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    csv_file = results_dir / "metrics.csv"

    if csv_file.exists():

        df = pd.read_csv(csv_file)

        df = df[df["Model"] != MODEL_NAME]

        df = pd.concat(
            [df, pd.DataFrame([result])],
            ignore_index=True
        )

    else:

        df = pd.DataFrame([result])

    df.to_csv(csv_file, index=False)

    df.to_excel(
        results_dir / "metrics.xlsx",
        index=False
    )

    print(df.sort_values(
        "Accuracy",
        ascending=False
    ))

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()