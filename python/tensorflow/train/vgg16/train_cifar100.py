from python.tensorflow.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/cifar100_png",
        output_file="resnet18_cifar100_tf.csv",
        checkpoint_path="checkpoints/resnet18_cifar100_tf.h5",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )
    
# PYTHONPATH=. python3 python/tensorflow/train/vgg16/train_cifar100.py