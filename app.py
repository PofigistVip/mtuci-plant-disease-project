import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

MODEL_NAME = "resnet50"
MODEL_PATH = "models/resnet50.pth"
DATASET_PATH = "data/split/train"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

classes = sorted(
    [
        p.name
        for p in Path(DATASET_PATH).iterdir()
        if p.is_dir()
    ]
)

@st.cache_resource
def load_model():

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(classes)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def save_history(prediction, confidence):

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    history_file = results_dir / "history.json"

    if history_file.exists():

        with open(
            history_file,
            "r",
            encoding="utf-8"
        ) as f:

            history = json.load(f)

    else:

        history = []

    history.append({
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=4
        )

st.set_page_config(
    page_title="Plant Disease Detector",
    layout="centered"
)

st.title("🌿 Определение болезни растения")

uploaded_file = st.file_uploader(
    "Загрузите изображение листа",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Загруженное изображение",
        use_container_width=True
    )

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = classes[
        predicted.item()
    ]

    confidence_percent = (
        confidence.item() * 100
    )

    st.success(
        f"Предсказание: {predicted_class}"
    )

    st.info(
        f"Вероятность: "
        f"{confidence_percent:.2f}%"
    )

    if st.button(
        "Сохранить результат"
    ):

        save_history(
            predicted_class,
            confidence_percent
        )

        st.success(
            "Результат сохранён в history.json"
        )

history_file = Path(
    "results/history.json"
)

if history_file.exists():

    st.subheader(
        "История запусков"
    )

    with open(
        history_file,
        "r",
        encoding="utf-8"
    ) as f:

        history = json.load(f)

    st.dataframe(
        history[::-1],
        use_container_width=True
    )