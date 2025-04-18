from python.pytorch.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/cifar100_png",
        output_file="resnet18_cifar100.csv",
        checkpoint_path="checkpoints/resnet18_cifar100.pth",
        img_size=(32, 32),
        grayscale=False
    )
    
    
# PYTHONPATH=. python3 python/pytorch/train/resnet18/train_cifar100.py