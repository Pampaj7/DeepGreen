import torch
from torchvision import models
import sys

def export_resnet18(num_classes: int = 100, pretrained_weights = None):
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


if __name__ == "__main__":
    export_resnet18(**({} if len(sys.argv) == 1 else {"num_classes": int(sys.argv[1])})) # weights=models.ResNet18_Weights.IMAGENET1K_V1 #