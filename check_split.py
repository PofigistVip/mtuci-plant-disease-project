from pathlib import Path

base = Path("data/split")

for part in ["train", "val", "test"]:
    count = len(list((base / part).rglob("*.jpg")))
    print(part, count)