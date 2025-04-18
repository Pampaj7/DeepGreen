from python.pytorch.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/fashion_mnist_png",
        output_file="resnet18_fashion.csv",
        checkpoint_path="checkpoints/resnet18_fashion.pth",
        img_size=(32, 32),
        grayscale=True
    )

# PYTHONPATH=. python3 python/pytorch/train/resnet18/train_fashion.py