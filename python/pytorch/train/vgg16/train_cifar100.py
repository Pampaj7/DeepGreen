from python.pytorch.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/cifar100_png",
        output_file="vgg16_cifar100.csv",
        checkpoint_path="checkpoints/vgg16_cifar100.pth",
        img_size=(32, 32),
        grayscale=False
    )