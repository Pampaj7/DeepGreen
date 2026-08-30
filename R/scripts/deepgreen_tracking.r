# Bridge from R to the shared measurement helper, tools/deepgreen_tracker.py.
#
# Replaces R/scripts/energy_tracking.r + R/scripts/tracker_control.py, which
# built their own EmissionsTracker with tracking_mode = "process" -- the only
# stack in the campaign to do so. Process mode excludes almost all host energy,
# so this stack's totals were not comparable with the other seven: its RAM
# energy share was 0.03% against ~50% elsewhere.
#
# Run contract (environment): DEEPGREEN_RUN_DIR, _ECOSYSTEM, _MODEL, _DATASET,
# _REP, _SEED, _EPOCHS, _DATA, _MODELS, _PYTHON.

if (!requireNamespace("processx", quietly = TRUE)) {
  stop("install.packages('processx')")
}

.dg <- new.env(parent = emptyenv())

dg_repo_root <- function() {
  # this file lives at <repo>/R/scripts/
  normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "R/scripts/x"), "..", ".."),
                mustWork = FALSE)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

dg_python <- function() {
  p <- Sys.getenv("DEEPGREEN_PYTHON", unset = "")
  if (nzchar(p) && file.exists(p)) return(p)
  venv <- file.path(getwd(), ".venv-deepgreen", "bin", "python")
  if (file.exists(venv)) return(venv)
  stop("no measurement interpreter: set DEEPGREEN_PYTHON. Refusing to fall back to ",
       "PATH python, which would make the CodeCarbon version an ambient property ",
       "of the shell.")
}

dg_run_params <- function() {
  as_num <- function(k, d) {
    v <- Sys.getenv(k, unset = "")
    if (nzchar(v)) as.numeric(v) else d
  }
  rep <- as_num("DEEPGREEN_REP", 0)
  list(repetition = rep,
       seed       = as_num("DEEPGREEN_SEED", 1000 + rep),
       epochs     = as_num("DEEPGREEN_EPOCHS", 30))
}

# Refuse to train a network that is not the one the study is comparing.
#
# The campaign driver puts models/MANIFEST.json's parameter count in
# DEEPGREEN_EXPECTED_PARAMS, so every stack can check it without a JSON parser
# of its own. R cannot join the shared-module group, for the reason recorded in
# R/models/resnet18.r: in torch 0.17.0 a script_module's $train() and $eval()
# raise "unused argument", so a loaded module cannot be switched out of training
# mode and would evaluate with batch norm on batch statistics. Its network is
# therefore rebuilt from torchvision-for-R, and this is what confirms the
# rebuild landed in the same place. VGG-16 ran as four different networks across
# the seven stacks, spanning 9.1x in parameters, under a specification claiming
# the counts were checked. They were not, anywhere.
dg_assert_params <- function(model) {
  want <- Sys.getenv("DEEPGREEN_EXPECTED_PARAMS", unset = "")
  if (!nzchar(want)) return(invisible(NULL))
  got <- sum(vapply(model$parameters, function(p) prod(dim(p)), numeric(1)))
  want <- as.numeric(want)
  if (got != want) {
    msg <- sprintf(paste0(
      "this stack has %.0f parameters; models/MANIFEST.json says %.0f ",
      "(difference %+.0f). Comparing its energy with the others would compare ",
      "models rather than ecosystems."), got, want, got - want)
    if (identical(Sys.getenv("DEEPGREEN_ALLOW_MODEL_DRIFT"), "1")) {
      cat("[deepgreen] WARNING", msg, "\n")
    } else {
      stop(msg, " Set DEEPGREEN_ALLOW_MODEL_DRIFT=1 to override.")
    }
  } else {
    cat(sprintf("[deepgreen] %.0f parameters, as exported\n", got))
  }
  invisible(got)
}

dg_init <- function() {
  run_dir <- Sys.getenv("DEEPGREEN_RUN_DIR", unset = "")
  if (!nzchar(run_dir)) {
    stop("DEEPGREEN_RUN_DIR is not set; the campaign driver provides it.")
  }
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

  bridge <- file.path(getwd(), "tools", "deepgreen_tracker.py")
  if (!file.exists(bridge)) stop("shared bridge not found at ", bridge)

  .dg$proc <- processx::process$new(
    dg_python(), c(bridge, "--daemon"),
    stdin = "|", stdout = "|", stderr = "|", env = c("current")
  )
  .dg$wait_for("tracker ready", timeout = 60)
  invisible(TRUE)
}

.dg$wait_for <- function(pattern, timeout = 30) {
  deadline <- Sys.time() + timeout
  repeat {
    line <- .dg$proc$read_output_lines(n = 1)
    if (length(line) && nzchar(line)) {
      if (grepl(pattern, line, fixed = TRUE)) return(invisible(line))
    } else {
      Sys.sleep(0.05)
    }
    if (Sys.time() > deadline) {
      stop("measurement bridge did not answer '", pattern, "' within ", timeout, "s")
    }
  }
}

# START and STOP are synchronous: the tracked window must open before the work
# begins. The Rust stack fired the command and started computing immediately,
# and its inference blocks recorded 0 J as a result.
dg_start <- function(phase, epoch) {
  stopifnot(phase %in% c("train", "eval"))
  .dg$proc$write_input(sprintf("START %s %d\n", phase, epoch))
  .dg$wait_for("START", timeout = 120)
  invisible(TRUE)
}

dg_stop <- function() {
  .dg$proc$write_input("STOP\n")
  .dg$wait_for("STOP", timeout = 120)
  invisible(TRUE)
}

# Per-epoch model quality. Every ecosystem computed accuracy in the first
# campaign and every one only printed it.
dg_metric <- function(epoch, train_loss, test_loss, test_acc) {
  .dg$proc$write_input(sprintf(
    "METRIC epoch=%d train_loss=%.6f test_loss=%.6f test_acc=%.4f\n",
    epoch, train_loss, test_loss, test_acc))
  .dg$wait_for("METRIC", timeout = 60)
  invisible(TRUE)
}

dg_shutdown <- function() {
  if (!is.null(.dg$proc) && .dg$proc$is_alive()) {
    try(.dg$proc$write_input("EXIT\n"), silent = TRUE)
    try(.dg$proc$wait(timeout = 30000), silent = TRUE)
    try(.dg$proc$kill(), silent = TRUE)
  }
  invisible(TRUE)
}
