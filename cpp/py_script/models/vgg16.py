import torch
from torchvision import models
import sys

def export_vgg16(output_name = "vgg16", num_classes: int = 100, pretrained_weights = None):
    model = models.vgg16(weights=pretrained_weights)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    # if not pretrained_weights:
    #     def init_weights(m):
    #         if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #             if m.bias is not None:
    #                 torch.nn.init.zeros_(m.bias)

    #     model.apply(init_weights)

    # # print details
    # print(model)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {trainable_params}")

    traced_model = torch.jit.script(model)
    traced_model.save(f"{output_name}.pt")


if __name__ == "__main__":
    params = {}
    if len(sys.argv) > 1:
        params["output_name"] = sys.argv[1]
    if len(sys.argv) > 2:
        params["num_classes"] = int(sys.argv[2])
    #params["pretrained_weights"] = models.VGG16_Weights.IMAGENET1K_V1

    export_vgg16(**(params))