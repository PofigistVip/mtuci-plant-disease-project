from pathlib import Path
import splitfolders

input_folder = "data/PlantVillage/color"
output_folder = "data/split"

splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.7, 0.15, 0.15),
    group_prefix=None
)

print("Готово!")