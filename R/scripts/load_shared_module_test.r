# Smoke test for spec S1: the shared TorchScript module must load into R/torch
# and produce the expected output shape.
#
# Run after scripts/export_torchscript_models.py:
#
#   Rscript R/scripts/load_shared_module_test.r
#
# The R torch package bundles its own LibTorch. A module exported by a torch
# newer than that bundle will fail here rather than mid-campaign.

suppressPackageStartupMessages(library(torch))

models_root <- Sys.getenv("DEEPGREEN_MODELS", unset = "models")

cases <- list(
  list("resnet18", "fashionmnist", 10L),
  list("resnet18", "cifar100", 100L),
  list("resnet18", "tinyimagenet200", 200L),
  list("vgg16", "fashionmnist", 10L),
  list("vgg16", "cifar100", 100L),
  list("vgg16", "tinyimagenet200", 200L)
)

cat("R torch package:", as.character(packageVersion("torch")), "\n")
cat("models root:", models_root, "\n\n")

failures <- 0L
for (cs in cases) {
  arch <- cs[[1]]; dataset <- cs[[2]]; num_classes <- cs[[3]]
  path <- file.path(models_root, paste0(arch, "_", dataset, ".pt"))
  ok <- tryCatch({
    m <- jit_load(path)
    out <- m(torch_zeros(c(2L, 3L, 32L, 32L)))
    shape <- as.integer(out$shape)
    good <- identical(shape, c(2L, num_classes))
    cat(sprintf("  %-8s %-16s out [%s]  %s\n", arch, dataset,
                paste(shape, collapse = ", "),
                if (good) "ok" else "SHAPE MISMATCH"))
    good
  }, error = function(e) {
    cat(sprintf("  %-8s %-16s FAILED: %s\n", arch, dataset, conditionMessage(e)))
    FALSE
  })
  if (!isTRUE(ok)) failures <- failures + 1L
}

if (failures > 0L) {
  cat(sprintf("\n%d module(s) failed; check models/MANIFEST.txt against the R torch bundle\n", failures))
  quit(status = 1L)
}
cat("\nall 6 shared modules load and forward in R/torch\n")
