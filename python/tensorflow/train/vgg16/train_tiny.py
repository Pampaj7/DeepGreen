from python.tensorflow.models.vgg16 import run_experiment

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/tiny_imagenet_png",
        output_file_train="vgg16_tiny_train.csv",
        output_file_eval="vgg16_tiny_eval.csv",
        checkpoint_path="checkpoints/vgg16_tiny_tf.h5",
        img_size=(32, 32),
        epochs=30,
        batch_size=128
    )

   # PYTHONPATH=. python3 python/tensorflow/train/vgg16/train_tiny.py