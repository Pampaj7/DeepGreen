"""Imagenette (Howard, fast.ai): ten ImageNet classes at native resolution.

The accelerator-saturation cell of the campaign. Every other dataset is
resized once, offline, to 32x32 (see download_convert_tinyimage.py); this one
is resized once, offline, to 224x224 by the standard ImageNet recipe --
shorter side to 224, centre crop -- so that no stack resizes anything at run
time and the seven stacks decode identical pixels, as specification S3
requires. Layout matches the other datasets: <root>/{train,test}/<class>/*.png.
"""
import os
import sys
import tarfile
import urllib.request

from PIL import Image
from tqdm import tqdm

TRAIN_RESOLUTION = (224, 224)
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

#: WordNet id -> class name, in Imagenette's canonical order.
CLASSES = {
    "n01440764": "tench",
    "n02102040": "english springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "french horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def download_and_extract(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    tgz = os.path.join(dest_dir, "imagenette2-320.tgz")
    root = os.path.join(dest_dir, "imagenette2-320")
    if not os.path.exists(tgz):
        print("Downloading Imagenette (320px)...")
        urllib.request.urlretrieve(URL, tgz)
    if not os.path.exists(root):
        print("Extracting...")
        with tarfile.open(tgz) as tf:
            tf.extractall(dest_dir)
    return root


def resize_centre_crop(img, size):
    w, h = img.size
    short = min(w, h)
    scale = size / short
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def convert(root, output_root):
    assert TRAIN_RESOLUTION[0] == TRAIN_RESOLUTION[1]
    for src_split, dst_split in (("train", "train"), ("val", "test")):
        print(f"Converting {src_split} -> {dst_split}...")
        for wnid, name in CLASSES.items():
            src = os.path.join(root, src_split, wnid)
            dst = os.path.join(output_root, dst_split, name)
            os.makedirs(dst, exist_ok=True)
            for fname in tqdm(sorted(os.listdir(src)), desc=name, leave=False):
                out = os.path.join(dst, os.path.splitext(fname)[0] + ".png")
                if os.path.exists(out):
                    continue
                with Image.open(os.path.join(src, fname)) as img:
                    resize_centre_crop(img.convert("RGB"), TRAIN_RESOLUTION[0]).save(out)


if __name__ == "__main__":
    data = os.environ.get("DEEPGREEN_DATA", "data")
    root = download_and_extract(data)
    convert(root, os.path.join(data, "imagenette_png"))
    for split in ("train", "test"):
        n = sum(len(os.listdir(os.path.join(data, "imagenette_png", split, c))) for c in CLASSES.values())
        print(f"{split}: {n} images")
