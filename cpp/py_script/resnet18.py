import torch
from torchvision import models

num_classes = 100
pretrained_weights = None # weights=models.ResNet18_Weights.IMAGENET1K_V1 #

for model, name in (
        (models.resnet18(weights=pretrained_weights), "resnet18"),
):
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if not pretrained_weights:
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)

    traced_model = torch.jit.script(model)
    traced_model.save(f"{name}.pt")