from dataset import get_dataloaders

_, _, _, classes = get_dataloaders()

print("Количество классов:", len(classes))
print(classes[:5])