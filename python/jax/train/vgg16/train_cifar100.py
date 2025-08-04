from python.jax.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/cifar100_png",
        output_file_base="vgg16_cifar100",
        checkpoint_path="checkpoints/vgg16_cifar100_jax.flax",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )

# PYTHONPATH=. python3 python/jax/train/vgg16/train_cifar100.py