from python.tensorflow.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/cifar100_png",
        output_file_train="resnet18_cifar100_train.csv",
        output_file_eval="resnet18_cifar100_eval.csv",
        checkpoint_path="checkpoints/resnet18_cifar100_tf.h5",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )

    # PYTHONPATH=. python3 python/tensorflow/train/resnet18/train_cifar100.py