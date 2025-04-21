from python.tensorflow.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/tiny_imagenet_png",
        output_file="resnet18_tiny_tf.csv",
        checkpoint_path="checkpoints/resnet18_tiny_tf.h5",
        img_size=(64, 64),
        epochs=30,
        batch_size=128
    )