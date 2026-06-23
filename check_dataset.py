from pathlib import Path

dataset = Path("data/PlantVillage/color")

classes = [d for d in dataset.iterdir() if d.is_dir()]

print("Количество классов:", len(classes))

total = 0
for cls in classes:
    total += len(list(cls.glob("*")))

print("Количество изображений:", total)