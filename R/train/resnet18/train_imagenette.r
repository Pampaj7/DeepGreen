library(torch)
library(torchvision)

source("R/models/resnet18.r")

cat("Script avviato!\n")

# Saturation cell (results/analysis/experiment_spec.md, "Why"): real
# ImageNet-resolution images so the accelerator is actually loaded, instead of
# 32x32 where the GPU sits at single-digit utilisation. Batch size is smaller
# than the main campaign's because VGG-16 at 224x224 does not fit an RTX
# 3090's 24 GB at batch 128 -- one named constant so the checker can grep it.
BATCH_SIZE <- 32
# The driver may raise loader parallelism for this cell; never hard-code 2.
LOADER_THREADS <- as.integer(Sys.getenv("DEEPGREEN_LOADER_THREADS", unset = "2"))

run_experiment(
    dataset_path = "data/imagenette_png",
    checkpoint_path = "R/checkpoints/resnet18_imagenette_r.pt",
    img_size = c(224, 224),
    grayscale = FALSE, # RGB, 3 channels
    test_split = "test",
    epochs = 30,
    batch_size = BATCH_SIZE,
    # PNGs are already 224x224 (resized once, offline, by
    # dataloader/download_convert_imagenette.py). transform_resize's
    # two-element `size` branch resamples unconditionally even when the
    # target size already matches, so this stack must skip it here to honour
    # "no stack resizes anything at run time" (spec S3).
    skip_resize = TRUE,
    num_workers = LOADER_THREADS
)
