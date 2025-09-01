# R/scripts/energy_tracking.r
if (!requireNamespace("processx", quietly = TRUE)) {
  stop("Installa 'processx' (install.packages(\"processx\"))")
}

.energy <- new.env(parent = emptyenv())
.energy$PYTHON_BIN   <- Sys.getenv("PYTHON_BIN", unset = "python")
.energy$backend      <- "daemon"     # "daemon" | "cli"
.energy$DAEMON_PATH  <- NULL
.energy$CLI_PATH     <- NULL
.energy$emissions_dir <- "emissions"
.energy$run_prefix   <- NULL
.energy$proc         <- NULL  # usato solo per backend=daemon

.detect_python <- function(python_bin) {
  cands <- c(python_bin, Sys.getenv("PYTHON_BIN", ""), "python3", "python")
  cands <- cands[nzchar(cands)]
  found <- Sys.which(cands); found <- found[found != ""]
  if (!length(found)) stop("Nessun interprete Python trovato. Imposta PYTHON_BIN.")
  unname(found[[1]])
}

.resolve_first <- function(cands) {
  for (p in cands) {
    if (file.exists(p)) return(normalizePath(p))
  }
  return(NULL)
}

# ---------- API PUBBLICA ----------

energy_init <- function(model, dataset, run_id = NULL,
                        emissions_dir = "emissions",
                        python_bin = Sys.getenv("PYTHON_BIN", unset = "python"),
                        backend = c("daemon","cli"),
                        daemon_path = NULL,
                        cli_path = NULL) {

  backend <- match.arg(backend)
  .energy$backend <- backend
  .energy$PYTHON_BIN <- .detect_python(python_bin)
  .energy$emissions_dir <- emissions_dir
  dir.create(emissions_dir, showWarnings = FALSE, recursive = TRUE)

  .energy$run_prefix <- sprintf("%s_%s", model, dataset)


  # risolvi path scripts
  if (is.null(daemon_path)) {
    .energy$DAEMON_PATH <- .resolve_first(c(
      "R/scripts/tracker_daemon.py",
      "scripts/tracker_daemon.py",
      "tracker_daemon.py"
    ))
  } else {
    .energy$DAEMON_PATH <- normalizePath(daemon_path)
  }

  if (is.null(cli_path)) {
    .energy$CLI_PATH <- .resolve_first(c(
      "R/scripts/run_tracker.py",
      "scripts/run_tracker.py",
      "run_tracker.py"
    ))
  } else {
    .energy$CLI_PATH <- normalizePath(cli_path)
  }

  if (backend == "daemon") {
    if (is.null(.energy$DAEMON_PATH)) {
      stop("Daemon non trovato (tracker_daemon.py). Passa daemon_path o metti il file in scripts/.")
    }
    # avvia daemon
    .energy$proc <- processx::process$new(
      .energy$PYTHON_BIN, c("-u", .energy$DAEMON_PATH),
      stdin="|", stdout="|", stderr="|", supervise=TRUE
    )
    # attende Ready
    t0 <- Sys.time()
    repeat {
      if (!.energy$proc$is_alive()) stop("Daemon non si avvia: ", .energy$proc$read_all_error())
      out <- .energy$proc$read_output_lines()
      if (any(grepl("\\[Daemon\\] Ready", out))) break
      if (as.numeric(difftime(Sys.time(), t0, units="secs")) > 5) stop("Timeout avvio daemon")
      Sys.sleep(0.05)
    }
  } else { # cli
    if (is.null(.energy$CLI_PATH)) {
      stop("CLI run_tracker.py non trovato. Passa cli_path o metti il file in scripts/.")
    }
  }

  invisible(TRUE)
}

energy_start_epoch <- function(phase, epoch) {
  stopifnot(phase %in% c("train","eval"))
  out_dir  <- normalizePath(.energy$emissions_dir)
  out_file <- sprintf("%s_%s_epoch%03d.csv", .energy$run_prefix, phase, epoch)

  if (.energy$backend == "daemon") {
    .write_wait_daemon(sprintf("START %s %s", out_dir, out_file),
                       wait_for="\\[Daemon\\] STARTED", timeout=10)
  } else {
    res <- processx::run(.energy$PYTHON_BIN,
                         c(.energy$CLI_PATH, "start",
                           "--output_dir", out_dir,
                           "--output_file", out_file),
                         echo = FALSE, error_on_status = FALSE)
    if (res$status != 0) {
      stop("run_tracker.py start failed: ", res$stderr, " | ", res$stdout)
    }
  }
  invisible(out_file)
}

energy_stop_epoch <- function() {
  if (.energy$backend == "daemon") {
    out <- .write_wait_daemon("STOP", wait_for="\\[Daemon\\] STOPPED", timeout=30)
    line <- tail(grep("\\[Daemon\\] STOPPED", out, value = TRUE), 1)
    return(as.numeric(sub(".*STOPPED\\s+([0-9\\.eE+-]+).*", "\\1", line)))
  } else {
    res <- processx::run(.energy$PYTHON_BIN,
                         c(.energy$CLI_PATH, "stop"),
                         echo = FALSE, error_on_status = FALSE)

    if (res$status != 0) {
      stop("run_tracker.py stop failed: ", res$stderr, " | ", res$stdout)
    }

    # 1) preferisci riga machine-readable EMISSIONS:<val> (vedi patch sotto)
    m1 <- regexec("EMISSIONS[: ]\\s*([0-9eE\\.+-]+)", res$stdout)
    mm1 <- regmatches(res$stdout, m1)[[1]]
    if (length(mm1) >= 2) return(as.numeric(mm1[2]))

    # 2) fallback al testo "[CodeCarbon] ... Emissions: X kg"
    m2 <- regexec("Emissions:\\s*([0-9eE\\.+-]+)\\s*kg", res$stdout)
    mm2 <- regmatches(res$stdout, m2)[[1]]
    if (length(mm2) >= 2) return(as.numeric(mm2[2]))

    # 3) fallback estremo: prendi la prima cifra con notazione scientifica o decimale
    m3 <- regexec("([0-9]+\\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", res$stdout)
    mm3 <- regmatches(res$stdout, m3)[[1]]
    if (length(mm3) >= 2) return(as.numeric(mm3[2]))

    return(NA_real_)
  }
}


energy_shutdown <- function() {
  if (.energy$backend == "daemon" && !is.null(.energy$proc)) {
    if (.energy$proc$is_alive()) {
      # tenta exit pulito
      try(.write_wait_daemon("EXIT", wait_for="\\[Daemon\\] BYE", timeout=3), silent=TRUE)
      if (.energy$proc$is_alive()) .energy$proc$kill()
    }
    .energy$proc <- NULL
  }
  invisible(TRUE)
}

# ---------- helpers backend=daemon ----------
.write_wait_daemon <- function(cmd, wait_for, timeout = 10) {
  p <- .energy$proc
  if (is.null(p) || !p$is_alive()) stop("Daemon non attivo")
  p$write_input(paste0(cmd, "\n"))
  t0 <- Sys.time()
  acc <- character()
  repeat {
    acc <- c(acc, p$read_output_lines())
    err_lines <- grep("\\[Daemon\\] ERROR", acc, value = TRUE)
    if (length(err_lines)) stop("Daemon error: ", tail(err_lines, 1))
    if (any(grepl(wait_for, acc))) break
    if (as.numeric(difftime(Sys.time(), t0, units="secs")) > timeout) {
      stop("Timeout su: ", wait_for, " (cmd=", cmd, "). Ultime righe: ",
           paste(tail(acc, 5), collapse=" | "))
    }
    Sys.sleep(0.02)
  }
  acc
}
