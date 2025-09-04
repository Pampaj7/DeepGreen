source("R/models/vgg16.r")
cat("Script avviato!\n")
run_experiment(
    dataset_path = "data/fashion_mnist_png",
    checkpoint_path = file.path(root, "R/checkpoints/vgg16_fashion_r.pt"),
    img_size = c(32, 32), # 32x32 anche per Fashion
    grayscale = TRUE, # convertirà a 3 canali
    test_split = "test",
    epochs = 30,
    batch_size = 128
)