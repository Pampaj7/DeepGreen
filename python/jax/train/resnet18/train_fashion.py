from python.jax.models.resnet18 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/fashion_mnist_png",
        output_file="resnet18_fashion_jax.csv",
        checkpoint_path="checkpoints/resnet18_fashion_jax.pkl",
        img_size=(32, 32),
        epochs=10,
        batch_size=128
    )
    
# PYTHONPATH=. python3 python/jax/train/resnet18/train_fashion.py