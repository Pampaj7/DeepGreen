library(torch)
library(torchvision)

source("R/models/resnet18.r")

cat("Script avviato!\n")

run_experiment(
    dataset_path = "data/fashion_mnist_png", 
    checkpoint_path = "R/checkpoints/resnet18_fashion_r.pt",
    img_size = c(32, 32), 
    grayscale = TRUE, # Fashion-MNIST è monocromatico → lo forziamo a 3 canali
    test_split = "test"
)