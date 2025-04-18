from python.pytorch.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/tiny_imagenet_png",
        output_file="vgg16_tiny.csv",
        checkpoint_path="checkpoints/vgg16_tiny.pth",
        img_size=(64, 64),
        grayscale=False,
        test_split="val"
    )