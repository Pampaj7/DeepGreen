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

# (architecture, dataset, classes, input resolution, batch). The last two exist
# because the accelerator-saturation cell (spec S7) runs at 224x224: a smoke test
# that only ever forwards 32x32 says nothing about the two modules that cell
# loads, and "all six modules load" was still the claim after the export started
# producing eight.
cases <- list(
  list("resnet18", "fashionmnist", 10L, 32L, 2L),
  list("resnet18", "cifar100", 100L, 32L, 2L),
  list("resnet18", "tinyimagenet200", 200L, 32L, 2L),
  list("resnet18", "imagenette", 10L, 224L, 1L),
  list("vgg16", "fashionmnist", 10L, 32L, 2L),
  list("vgg16", "cifar100", 100L, 32L, 2L),
  list("vgg16", "tinyimagenet200", 200L, 32L, 2L),
  list("vgg16", "imagenette", 10L, 224L, 1L)
)

cat("R torch package:", as.character(packageVersion("torch")), "\n")
cat("models root:", models_root, "\n\n")

failures <- 0L
for (cs in cases) {
  arch <- cs[[1]]; dataset <- cs[[2]]; num_classes <- cs[[3]]
  px <- cs[[4]]; batch <- cs[[5]]
  path <- file.path(models_root, paste0(arch, "_", dataset, ".pt"))
  ok <- tryCatch({
    m <- jit_load(path)
    out <- m(torch_zeros(c(batch, 3L, px, px)))
    shape <- as.integer(out$shape)
    good <- identical(shape, c(batch, num_classes))
    cat(sprintf("  %-8s %-16s in %dx%d  out [%s]  %s\n", arch, dataset, px, px,
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
cat(sprintf("\nall %d shared modules load and forward in R/torch\n", length(cases)))
