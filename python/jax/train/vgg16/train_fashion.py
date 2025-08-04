from python.jax.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/fashion_mnist_png",
        output_file_base="vgg16_fashion",
        checkpoint_path="checkpoints/vgg16_fashion_jax.flax",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )

# PYTHONPATH=. python3 python/jax/train/vgg16/train_fashion.py