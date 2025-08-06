import os
import json
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

def save_image(img_tensor, path):
    img = transforms.ToPILImage()(img_tensor)
    img.save(path)

# be careful on remote machine you probably need to change path

def convert_fashionmnist_to_png(output_root="/home/pampaj/DeepGreen/data/fashion_mnist_png"):
    try:
        os.makedirs(output_root, exist_ok=True)
    except Exception as e:
        print("Errore nella creazione della cartella:", e)

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_data = FashionMNIST(root="./data", train=False, download=True, transform=transform)

    label_map = {idx: name for idx, name in enumerate(train_data.classes)}
    with open(os.path.join(output_root, "classes.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    def export_split(dataset, split):
        print(f"Exporting {split}...")
        for idx in tqdm(range(len(dataset))):
            img, label = dataset[idx]
            class_name = label_map[label]
            class_dir = os.path.join(output_root, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            img_filename = f"{idx:05d}.png"
            save_image(img, os.path.join(class_dir, img_filename))

    export_split(train_data, "train")
    export_split(test_data, "test")


if __name__ == "__main__":
    convert_fashionmnist_to_png(**({} if len(sys.argv) == 1 else {"output_root": sys.argv[1]}))
