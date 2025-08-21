source("R/models/vgg16.r")
cat("Script avviato!\n")
run_experiment(
    dataset_path = dataset_path,
    checkpoint_path = file.path(root, "R/checkpoints/vgg16_tiny_r.pt"),
    img_size = c(32, 32), # forziamo 32x32 anche per Tiny (consistenza)
    grayscale = FALSE,
    test_split = "val", # Tiny usa 'val' come split
    epochs = 30,
    batch_size = 128
)