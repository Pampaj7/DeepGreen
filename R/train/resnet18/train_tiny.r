library(torch)
library(torchvision)

source("R/models/resnet18.r")

cat("Script avviato!\n")

run_experiment(
    dataset_path = "data/tiny_imagenet_png", 
    checkpoint_path = "R/checkpoints/resnet18_tiny_r.pt",
    img_size = c(32, 32), # Tiny ImageNet di default è 64×64
    grayscale = FALSE, # RGB a 3 canali
    test_split = "val", # Tiny ImageNet ha 'train' e 'val'
    epochs = 30,
    batch_size = 128
)