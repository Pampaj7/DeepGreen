from python.pytorch.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/fashion_mnist_png",
        output_file="vgg16_fashion.csv",
        checkpoint_path="checkpoints/vgg16_fashion.pth",
        img_size=(32, 32),
        grayscale=True
    )