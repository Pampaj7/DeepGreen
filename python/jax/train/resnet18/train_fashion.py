from python.jax.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/fashion_mnist_png",
        output_file_base="resnet18_fashion",
        checkpoint_path="checkpoints/resnet18_fashion_jax.flax",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )

# PYTHONPATH=. python3 python/jax/train/resnet18/train_fashion.py
