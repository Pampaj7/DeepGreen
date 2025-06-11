import torch
from torchvision import models
import sys

def export_resnet18(output_name = "resnet18", num_classes: int = 100, pretrained_weights = None):
    model = models.resnet18(weights=pretrained_weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if not pretrained_weights:
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)

    traced_model = torch.jit.script(model)
    traced_model.save(f"{output_name}.pt")


if __name__ == "__main__":
    params = {}
    if len(sys.argv) > 1:
        params["output_name"] = sys.argv[1]
    if len(sys.argv) > 2:
        params["num_classes"] = int(sys.argv[2])
    #params["pretrained_weights"] = weights=models.ResNet18_Weights.IMAGENET1K_V1

    export_resnet18(**(params))