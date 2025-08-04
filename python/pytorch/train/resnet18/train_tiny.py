from python.pytorch.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/tiny_imagenet_png",
        output_file="resnet18_tiny.csv",
        checkpoint_path="checkpoints/resnet18_tiny.pth",
        img_size=(32, 32),
        grayscale=False,
        test_split="val"
    )

#PYTHONPATH=. python3 python/pytorch/train/resnet18/train_tiny.py