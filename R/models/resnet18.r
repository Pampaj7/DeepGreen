library(torch)
library(torchvision)

# Costruisci ResNet18
build_resnet18 <- function(num_classes = 100, pretrained = FALSE) {
  model <- model_resnet18(pretrained = pretrained)
  model$fc <- nn_linear(model$fc$in_features, num_classes)
  if (!pretrained) {
    model$apply(function(m) {
      if (inherits(m, "nn_conv2d") || inherits(m, "nn_linear")) {
        nn_init_kaiming_normal_(m$weight)
        if (!is.null(m$bias)) nn_init_zeros_(m$bias)
      }
    })
  }
  model
}

get_loaders <- function(dataset_path, batch_size = 128, img_size = c(32, 32), grayscale = FALSE, test_split = "test") {
  # torchvision R vuole size = c(height, width)
transform_list <- list(
  transform_resize(size = c(img_size[2], img_size[1]))
)
if (grayscale) {
  transform_list <- c(transform_list, transform_grayscale(num_output_channels = 3))
}
transform_list <- c(transform_list, transform_to_tensor())
transform <- transform_compose(transform_list)


  train_set <- image_folder_dataset(file.path(dataset_path, "train"), transform = transform)
  test_set <- image_folder_dataset(file.path(dataset_path, test_split), transform = transform)
  train_loader <- dataloader(train_set, batch_size = batch_size, shuffle = TRUE, num_workers = 2)
  test_loader <- dataloader(test_set, batch_size = batch_size, shuffle = FALSE, num_workers = 2)
  list(train_loader = train_loader, test_loader = test_loader, num_classes = length(train_set$classes))
}

# Train loop
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

# Eval loop
evaluate <- function(model, test_loader, criterion, device) {
  model$eval()
  total <- 0
  correct <- 0
  loss_sum <- 0
  torch_no_grad({
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
  acc <- 100 * correct / total
  list(loss = loss_sum / total, acc = acc)
}

# Esperimento
run_experiment <- function(dataset_path, checkpoint_path, img_size = c(32, 32), grayscale = FALSE, test_split = "test",
                           epochs = 30, batch_size = 128) {
  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  loaders <- get_loaders(dataset_path, batch_size, img_size, grayscale, test_split)
  model <- build_resnet18(num_classes = loaders$num_classes, pretrained = FALSE)
  model$to(device = device)
  criterion <- nn_cross_entropy_loss()
  optimizer <- optim_adam(model$parameters, lr = 1e-4)
  for (epoch in 1:epochs) {
    cat(sprintf("Epoch %d/%d\n", epoch, epochs))
    train_loss <- train(model, loaders$train_loader, criterion, optimizer, device)
    eval <- evaluate(model, loaders$test_loader, criterion, device)
    cat(sprintf("Train Loss=%.4f, Test Loss=%.4f, Test Acc=%.2f%%\n", train_loss, eval$loss, eval$acc))
  }
  dir.create("checkpoints", showWarnings = FALSE)
  torch_save(model$state_dict(), checkpoint_path)
}

# Esempio di chiamata:
# run_experiment("data/cifar100_png", "checkpoints/resnet18_cifar100_r.pt")