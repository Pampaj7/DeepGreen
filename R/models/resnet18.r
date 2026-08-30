# R/models/resnet18.r
library(torch)
library(torchvision)
library(coro)

# --- tracking utility (CodeCarbon via Python CLI) ---
# Shared measurement bridge (spec S5). The stack-private helper set
# tracking_mode = "process", the only one in the campaign to do so, which
# excluded almost all host energy and made its totals incomparable.
source("R/scripts/deepgreen_tracking.r")

# ===== Helpers (log) =====
.now <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
.log <- function(...) cat(sprintf("[%s] ", .now()), sprintf(...), "\n")

# ===== Modello =====
build_resnet18 <- function(num_classes = 100, pretrained = FALSE) {
  model <- torchvision::model_resnet18(pretrained = pretrained)
  in_f <- model$fc$in_features
  model$fc <- nn_linear(in_f, num_classes)
  model
}

# ===== Data loaders =====
get_loaders <- function(dataset_path, batch_size = 128, img_size = c(32, 32),
                        grayscale = FALSE, test_split = "test") {
  .log("Checking dataset path: %s", dataset_path)
  if (!dir.exists(file.path(dataset_path, "train"))) {
    stop("Train directory does not exist: ", file.path(dataset_path, "train"))
  }
  if (!dir.exists(file.path(dataset_path, test_split))) {
    stop("Test directory does not exist: ", file.path(dataset_path, test_split))
  }

  # Fashion MNIST is grayscale (1 channel), but ResNet18 expects 3 channels
  transform <- function(img) {
    # transform_to_tensor, as R/VGG-16 and every other stack does it. This read
    # `torch_tensor(img / 255)`, and image_folder_dataset's default loader
    # already returns values in [0, 1] -- measured: base_loader on a campaign
    # PNG gives an array with range 0 to 1. So R/ResNet-18 trained on [0, 1/255]
    # while R/VGG-16 trained on [0, 1]: one ecosystem, two input pipelines, and
    # S3 fixes the pipeline for all stacks with no check covering it. Largely
    # absorbed by the BatchNorm after conv1, which is why it never showed in the
    # accuracies -- and why nothing but reading the source would have found it.
    img <- torchvision::transform_to_tensor(img)

    # Resize to target size
    img <- torchvision::transform_resize(img, size = c(img_size[2], img_size[1]))

    # If grayscale, replicate the single channel to 3 channels
    if (grayscale) {
      if (img$size(1) == 1) {
        img <- img$repeat_interleave(3, dim = 1) # Replicate to [3, height, width]
      }
    }
    
    img
  }

  .log("Loading train dataset from %s", file.path(dataset_path, "train"))
  train_set <- tryCatch({
    torchvision::image_folder_dataset(file.path(dataset_path, "train"), transform = transform)
  }, error = function(e) {
    stop("Failed to load train dataset: ", e$message)
  })
  .log("Train dataset loaded. Classes: %s, Samples: %d", paste(train_set$classes, collapse = ", "), length(train_set))

  .log("Loading test dataset from %s", file.path(dataset_path, test_split))
  test_set <- tryCatch({
    torchvision::image_folder_dataset(file.path(dataset_path, test_split), transform = transform)
  }, error = function(e) {
    stop("Failed to load test dataset: ", e$message)
  })
  .log("Test dataset loaded. Classes: %s, Samples: %d", paste(test_set$classes, collapse = ", "), length(test_set))

  train_loader <- dataloader(train_set, batch_size = batch_size, shuffle = TRUE, num_workers = 2)  # was 0: single-threaded decoding made R ~11x slower than Rust
  test_loader  <- dataloader(test_set, batch_size = batch_size, shuffle = FALSE, num_workers = 2)  # was 0: single-threaded decoding made R ~11x slower than Rust

  list(
    train_loader = train_loader,
    test_loader = test_loader,
    num_classes = length(train_set$classes)
  )
}
# ===== Train / Eval =====
train <- function(model, train_loader, criterion, optimizer, device) {
  model$train()
  running_loss <- 0
  coro::loop(for (b in train_loader) {
    inputs <- b[[1]]$to(device = device)
    targets <- b[[2]]$to(device = device)
    optimizer$zero_grad()
    outputs <- model(inputs)
    loss <- criterion(outputs, targets)
    loss$backward()
    optimizer$step()
    running_loss <- running_loss + loss$item() * inputs$size(1)
  })
  running_loss / length(train_loader$dataset)
}

evaluate <- function(model, test_loader, criterion, device) {
  model$eval()
  total <- 0; correct <- 0; loss_sum <- 0
  with_no_grad({
    coro::loop(for (b in test_loader) {
      inputs <- b[[1]]$to(device = device)
      targets <- b[[2]]$to(device = device)
      outputs <- model(inputs)
      loss <- criterion(outputs, targets)
      loss_sum <- loss_sum + loss$item() * inputs$size(1)
      pred <- torch_max(outputs, dim = 2)[[2]]
      total <- total + targets$size(1)
      correct <- correct + (pred == targets)$sum()$item()
    })
  })
  list(loss = loss_sum / total, acc = 100 * correct / total)
}


# ===== Shared TorchScript module (spec S1) =====
# The four LibTorch-based ecosystems must train the SAME module. In the first
# campaign this stack built torchvision-for-R's own ResNet-18/VGG-16 while C++
# and Rust each used a hand-written port and Python used torchvision, so the
# "one backend, four bindings" control group actually compared four
# implementations. The module is produced once by
# scripts/export_torchscript_models.py; the exporting torch build must match the
# LibTorch that the R torch package links against.
deepgreen_model_path <- function(arch, dataset) {
  root <- Sys.getenv("DEEPGREEN_MODELS", unset = "models")
  file.path(root, paste0(arch, "_", dataset, ".pt"))
}

deepgreen_dataset_key <- function(dataset_path) {
  base <- basename(dataset_path)
  switch(sub("_png$", "", base),
         "cifar100"      = "cifar100",
         "fashion_mnist" = "fashionmnist",
         "tiny_imagenet" = "tinyimagenet200",
         stop(sprintf("unknown dataset directory: %s", base)))
}

load_shared_module <- function(arch, dataset_path, device) {
  path <- deepgreen_model_path(arch, deepgreen_dataset_key(dataset_path))
  if (!file.exists(path)) {
    stop(sprintf(
      "shared TorchScript module not found at %s; run scripts/export_torchscript_models.py",
      path))
  }
  m <- torch::jit_load(path)
  m$to(device = device)
  m
}

# ===== Esperimento (con tracking integrato) =====
run_experiment <- function(dataset_path, checkpoint_path,
                           img_size = c(32, 32), grayscale = FALSE, test_split = "test",
                           epochs = 30, batch_size = 128,
                           run_id = NULL, python_bin = Sys.getenv("PYTHON_BIN", unset = "python")) {

  params <- dg_run_params()
  epochs <- params$epochs
  torch_manual_seed(params$seed)
  set.seed(params$seed)

  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  if (device$type != "cuda") {
    stop("R/torch does not see a GPU. Measuring now would attribute a CPU fallback ",
         "to the ecosystem; set DEEPGREEN_ALLOW_CPU=1 to override deliberately.")
  }
  .log("device=%s | repetition=%s seed=%s epochs=%s",
       device$type, params$repetition, params$seed, epochs)

  loaders <- get_loaders(dataset_path, batch_size, img_size, grayscale, test_split)
  # R/torch is the one LibTorch binding that cannot use the shared TorchScript
  # module (spec S1). In torch 0.17.0 a script_module's $train() and $eval()
  # raise "unused argument", and the underlying handle is not reachable through
  # the documented fields, so the module cannot be switched between training and
  # evaluation. The shared modules are exported in training mode, so this stack
  # would evaluate with batch norm using batch statistics -- the same defect
  # found in the TensorFlow and Rust stacks. It therefore builds its own model,
  # and the architecture parity that holds for Python, C++ and Rust does not
  # hold here. See results/analysis/experiment_spec.md, S1.
  model <- build_resnet18(num_classes = loaders$num_classes, pretrained = FALSE)
  dg_assert_params(model)
  model$to(device = device)
  criterion <- nn_cross_entropy_loss()
  optimizer <- optim_adam(model$parameters, lr = 1e-4)

  # --- init energy tracker (CLI) ---
  dataset_name <- basename(dataset_path)  # es: "cifar100_png"
  dg_init()
  dg_datafp(loaders$test_loader)

  on.exit(dg_shutdown(), add = TRUE)

  # crea la cartella del checkpoint finale se manca
  dir.create(dirname(checkpoint_path), showWarnings = FALSE, recursive = TRUE)

  for (epoch in 1:epochs) {
    .log("Epoch %d/%d", epoch, epochs)

    # ---- TRAIN (tracciato) ----
    dg_start("train", epoch)
    train_loss <- train(model, loaders$train_loader, criterion, optimizer, device)
    dg_stop()

    # ---- EVAL (tracciato) ----
    dg_start("eval", epoch)
    eval <- evaluate(model, loaders$test_loader, criterion, device)
    dg_stop()

    # Outside the tracked window: writing the metric must not be measured.
    dg_metric(epoch, train_loss, eval$loss, eval$acc)
    .log("Train Loss=%.4f | Test Loss=%.4f, Acc=%.2f%%",
         train_loss, eval$loss, eval$acc)
  }

  # Salva solo il modello finale
  torch_save(model$state_dict(), checkpoint_path)
  .log("Modello finale salvato in %s", checkpoint_path)
}
