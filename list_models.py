import timm

models = timm.list_models(pretrained=True)

for m in models:
    if "resnet50" in m:
        print(m)