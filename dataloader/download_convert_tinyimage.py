import os
import zipfile
import urllib.request
from PIL import Image
import shutil
import json
from tqdm import tqdm
import sys


def download_and_extract_tiny_imagenet(dest_dir="./data"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(dest_dir, "tiny-imagenet-200.zip")
    extract_path = os.path.join(dest_dir, "tiny-imagenet-200")

    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(zip_path):  #TODO
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

    return extract_path

# be careful on remote machine you probably need to change path

def convert_tiny_imagenet_to_png(dataset_root="./data/tiny-imagenet-200", output_root="/home/pampaj/DeepGreen/data/tiny_imagenet_png"):
    os.makedirs(output_root, exist_ok=True)

    wnids_path = os.path.join(dataset_root, "wnids.txt")
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f.readlines()]

    # Map wnids to words (if available)
    words_path = os.path.join(dataset_root, "words.txt")
    label_map = {}
    with open(words_path, "r") as f:
        for line in f:
            wnid, label = line.strip().split('\t')
            if wnid in wnids:
                label_map[wnid] = label

    with open(os.path.join(output_root, "classes.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Process training images
    print("Converting training images...")
    train_dir = os.path.join(dataset_root, "train")
    for wnid in tqdm(wnids):
        images_dir = os.path.join(train_dir, wnid, "images")
        class_dir = os.path.join(output_root, "train", label_map[wnid])
        os.makedirs(class_dir, exist_ok=True)
        for i, img_file in enumerate(os.listdir(images_dir)):
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            img.save(os.path.join(class_dir, f"{wnid}_{i:05d}.png"))

    # Process validation images
    print("Converting validation images...")
    val_dir = os.path.join(dataset_root, "val")
    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
    val_img_dir = os.path.join(val_dir, "images")

    val_map = {}
    with open(val_annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            val_map[parts[0]] = parts[1]

    for img_file, wnid in tqdm(val_map.items()):
        label = label_map[wnid]
        class_dir = os.path.join(output_root, "val", label)
        os.makedirs(class_dir, exist_ok=True)
        src_path = os.path.join(val_img_dir, img_file)
        dst_path = os.path.join(class_dir, img_file.replace(".JPEG", ".png"))
        img = Image.open(src_path)
        img.save(dst_path)


if __name__ == "__main__":
    dataset_path = download_and_extract_tiny_imagenet()
    convert_tiny_imagenet_to_png(dataset_path, **({} if len(sys.argv) == 1 else {"output_root": sys.argv[1]}))
