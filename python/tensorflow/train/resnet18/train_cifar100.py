from python.tensorflow.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/cifar100_png",
        output_file="vgg16_cifar100_tf.csv",
        checkpoint_path="checkpoints/vgg16_cifar100_tf.h5",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )