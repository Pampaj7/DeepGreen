source("R/models/vgg16.r")
cat("Script avviato!\n")
run_experiment(
  dataset_path = "data/cifar100_png",
  checkpoint_path = file.path("R/checkpoints/vgg16_cifar100_r.pt"),
  img_size = c(32, 32),
  grayscale = FALSE,
  test_split = "test",
  epochs = 30,
  batch_size = 128
)