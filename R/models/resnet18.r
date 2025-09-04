# R/models/resnet18.r
library(torch)
library(torchvision)
library(coro)

# --- tracking utility (CodeCarbon via Python CLI) ---
source("R/scripts/energy_tracking.r")

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

  if (grayscale) {
    transform <- function(img) {
      img <- torch_tensor(img / 255, dtype = torch_float())$permute(c(3,1,2))
      img %>%
        torchvision::transform_resize(size = c(img_size[2], img_size[1])) %>%
        torchvision::transform_grayscale(num_output_channels = 3)
    }
  } else {
    transform <- function(img) {
      img <- torch_tensor(img / 255, dtype = torch_float())$permute(c(3,1,2))
      img %>% torchvision::transform_resize(size = c(img_size[2], img_size[1]))
    }
  }

  train_set <- torchvision::image_folder_dataset(file.path(dataset_path, "train"), transform = transform)
  test_set  <- torchvision::image_folder_dataset(file.path(dataset_path, test_split), transform = transform)

  train_loader <- dataloader(train_set, batch_size = batch_size, shuffle = TRUE,  num_workers = 2)
  test_loader  <- dataloader(test_set,  batch_size = batch_size, shuffle = FALSE, num_workers = 2)

  list(
    train_loader = train_loader,
    test_loader  = test_loader,
    num_classes  = length(train_set$classes)
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

# ===== Esperimento (con tracking integrato) =====
run_experiment <- function(dataset_path, checkpoint_path,
                           img_size = c(32, 32), grayscale = FALSE, test_split = "test",
                           epochs = 30, batch_size = 128,
                           run_id = NULL, python_bin = Sys.getenv("PYTHON_BIN", unset = "python")) {

  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  .log("Script avviato! device=%s", device$type)

  loaders <- get_loaders(dataset_path, batch_size, img_size, grayscale, test_split)
  model <- build_resnet18(num_classes = loaders$num_classes, pretrained = FALSE)
  model$to(device = device)
  criterion <- nn_cross_entropy_loss()
  optimizer <- optim_adam(model$parameters, lr = 1e-4)

  # --- init energy tracker (CLI) ---
  dataset_name <- basename(dataset_path)  # es: "cifar100_png"
  energy_init(
    model = "resnet18",
    dataset = dataset_name,
    run_id = run_id,
    emissions_dir = file.path("R", "emissions"),  # <<<<< path esplicito
    python_bin = python_bin,
    backend = "daemon"   # ⟵ invece di "cli"
  )

  on.exit(energy_shutdown(), add = TRUE)

  # crea la cartella del checkpoint finale se manca
  dir.create(dirname(checkpoint_path), showWarnings = FALSE, recursive = TRUE)

  for (epoch in 1:epochs) {
    .log("Epoch %d/%d", epoch, epochs)

    # ---- TRAIN (tracciato) ----
    energy_start_epoch("train", epoch)
    train_loss <- train(model, loaders$train_loader, criterion, optimizer, device)
    train_co2  <- energy_stop_epoch()

    # ---- EVAL (tracciato) ----
    energy_start_epoch("eval", epoch)
    eval <- evaluate(model, loaders$test_loader, criterion, device)
    eval_co2 <- energy_stop_epoch()

    .log("Train Loss=%.4f (CO2=%.6f kg) | Test Loss=%.4f, Acc=%.2f%% (CO2=%.6f kg)",
         train_loss, train_co2, eval$loss, eval$acc, eval_co2)
  }

  # Salva solo il modello finale
  torch_save(model$state_dict(), checkpoint_path)
  .log("Modello finale salvato in %s", checkpoint_path)
}
