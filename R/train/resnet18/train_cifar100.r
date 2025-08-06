source("R/models/resnet18.r")
cat("Script avviato!\n")
run_experiment(
  dataset_path = "/home/pampaj/DeepGreen/data/cifar100_png",
  checkpoint_path = "R/checkpoints/resnet18_cifar100_r.pt",
  img_size = c(32, 32),
  grayscale = FALSE
)