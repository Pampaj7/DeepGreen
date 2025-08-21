# R/models/vgg16.r
library(torch)
library(torchvision)

# -------------------- VGG16 (community-accepted) per 32x32 --------------------
build_vgg16 <- function(num_classes = 100, pretrained = FALSE) {
    model <- torchvision::model_vgg16(pretrained = pretrained)

    # Adatta per input 32x32: dopo 5 pool hai 1x1, quindi uso GAP 1x1 e classifier compatto
    model$avgpool <- nn_adaptive_avg_pool2d(output_size = c(1, 1))
    model$classifier <- nn_sequential(
        nn_flatten(), # 512*1*1 -> 512
        nn_linear(512, 512),
        nn_relu(),
        nn_dropout(p = 0.5),
        nn_linear(512, num_classes)
    )
    model
}

# -------------------- DATA LOADER (sempre 32x32) --------------------
get_loaders <- function(dataset_path, batch_size = 128, img_size = c(32, 32),
                        grayscale = FALSE, test_split = "test", num_workers = 0) {
    # pipeline robusta: to_tensor -> (flip) -> resize -> (grayscale)
    base_transform <- function(img) {
        x <- torchvision::transform_to_tensor(img) # [0,1], CHW
        x <- torchvision::transform_resize(x, size = c(img_size[2], img_size[1]))
        x
    }

    if (grayscale) {
        transform <- function(img) {
            x <- base_transform(img)
            x <- torchvision::transform_grayscale(x, num_output_channels = 3) # 3 canali
            x
        }
    } else {
        transform <- function(img) {
            x <- base_transform(img)
            # augment leggerissimo (opzionale)
            if (runif(1) < 0.5) x <- torchvision::transform_hflip(x)
            x
        }
    }

    train_set <- torchvision::image_folder_dataset(
        file.path(dataset_path, "train"),
        transform = transform
    )

    split <- if (file.exists(file.path(dataset_path, "test"))) "test" else test_split
    test_set <- torchvision::image_folder_dataset(
        file.path(dataset_path, split),
        transform = base_transform # no augment in eval
    )

    train_loader <- dataloader(train_set, batch_size = batch_size, shuffle = TRUE, num_workers = num_workers)
    test_loader <- dataloader(test_set, batch_size = batch_size, shuffle = FALSE, num_workers = num_workers)

    list(
        train_loader = train_loader,
        test_loader  = test_loader,
        num_classes  = length(train_set$classes)
    )
}

# -------------------- LOOP --------------------
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
    total <- 0
    correct <- 0
    loss_sum <- 0
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

# -------------------- RUN --------------------
run_experiment <- function(dataset_path, checkpoint_path, img_size = c(32, 32),
                           grayscale = FALSE, test_split = "test",
                           epochs = 30, batch_size = 128, num_workers = 0) {
    device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
    cat("Script avviato!\n")

    loaders <- get_loaders(dataset_path, batch_size, img_size, grayscale, test_split, num_workers)
    model <- build_vgg16(num_classes = loaders$num_classes, pretrained = FALSE)
    model$to(device = device)

    criterion <- nn_cross_entropy_loss()
    optimizer <- optim_adam(model$parameters, lr = 1e-4)

    for (epoch in 1:epochs) {
        cat(sprintf("Epoch %d/%d\n", epoch, epochs))
        train_loss <- train(model, loaders$train_loader, criterion, optimizer, device)
        eval <- evaluate(model, loaders$test_loader, criterion, device)
        cat(sprintf(
            "Train Loss=%.4f, Test Loss=%.4f, Test Acc=%.2f%%\n",
            train_loss, eval$loss, eval$acc
        ))
    }

    dir.create(dirname(checkpoint_path), showWarnings = FALSE, recursive = TRUE)
    torch_save(model$state_dict(), checkpoint_path)
}