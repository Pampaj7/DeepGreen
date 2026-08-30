use std::process::{Command, Child, ChildStdin, ChildStdout, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::{fs, thread, time::Duration};

struct TrackerState {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    /// Acknowledgements from the bridge. START and STOP must be synchronous:
    /// the first campaign wrote the command into the pipe and began computing
    /// immediately, so the tracked window did not cover the work it was meant to
    /// measure -- CodeCarbon takes seconds to initialise, and a short inference
    /// phase could finish before the tracker had started.
    stdout: Option<BufReader<ChildStdout>>,
    active: bool,
}

static TRACKER: Lazy<Mutex<TrackerState>> = Lazy::new(|| {
    Mutex::new(TrackerState {
        child: None,
        stdin: None,
        stdout: None,
        active: false,
    })
});

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate must live under the repository root")
        .to_path_buf()
}

fn get_daemon_script() -> PathBuf {
    // The shared bridge, so this stack is measured with the same CodeCarbon
    // configuration as every other. The first campaign used a private copy under
    // rust/scripts/ with its own settings.
    repo_root().join("tools").join("deepgreen_tracker.py")
}

/// Interpreter that runs the measurement bridge.
///
/// The first campaign spawned bare `python3`, so the CodeCarbon *version*
/// measuring this stack was whatever the ambient PATH resolved -- which is how
/// one campaign came to mix CodeCarbon 2.8.4 and 3.0.4 across ecosystems.
fn tracker_python() -> String {
    if let Ok(p) = std::env::var("DEEPGREEN_PYTHON") {
        return p;
    }
    let venv = repo_root().join(".venv-deepgreen").join("bin").join("python");
    if venv.exists() {
        return venv.to_string_lossy().into_owned();
    }
    panic!(
        "no measurement interpreter: set DEEPGREEN_PYTHON, or create .venv-deepgreen \
         with scripts/setup_environment.sh. Refusing to fall back to PATH python3, \
         which would make the CodeCarbon version an ambient property of the shell."
    )
}

pub fn init_tracker_daemon() {
    let script_path = get_daemon_script();
    println!("[Rust] Launching tracker daemon: {:?}", script_path);

    let mut child = Command::new(tracker_python())
        .arg(script_path)
        .arg("--daemon")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("❌ Failed to start tracker daemon");

    let stdin = child.stdin.take().expect("failed to get stdin for the tracker");
    let stdout = child.stdout.take().expect("failed to get stdout for the tracker");
    let mut reader = BufReader::new(stdout);

    // Block until the bridge reports it is ready, so no work is done before the
    // measurement machinery exists.
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("tracker bridge died before signalling readiness");
    println!("[Rust] {}", line.trim());

    let mut state = TRACKER.lock().unwrap();
    state.child = Some(child);
    state.stdin = Some(stdin);
    state.stdout = Some(reader);
    state.active = false;

    println!(
        "[Rust] Tracker daemon started (PID: {:?})",
        state.child.as_ref().unwrap().id()
    );
}

pub fn start_tracker(phase: &str, epoch: u32) {
    // fix per lock di CodeCarbon
    let _ = fs::remove_file("/tmp/.codecarbon.lock");

    let mut state = TRACKER.lock().unwrap();
    if state.active {
        eprintln!("tracker already active, ignoring START {} {}", phase, epoch);
        return;
    }

    if let Some(stdin) = state.stdin.as_mut() {
        writeln!(stdin, "START {} {}", phase, epoch)
            .expect("failed to write START to the tracker");
        stdin.flush().unwrap();
        state.active = true;
    } else {
        panic!("tracker bridge not initialised");
    }
    // Wait for the acknowledgement: the tracked window must open before the
    // workload starts, not at some point during it.
    if let Some(reader) = state.stdout.as_mut() {
        let mut ack = String::new();
        reader.read_line(&mut ack).expect("tracker bridge died on START");
        println!("[Rust] {}", ack.trim());
    } else {
        panic!("❌ Tracker daemon not initialized");
    }
}
pub fn stop_tracker() {
    let mut state = TRACKER.lock().unwrap();
    if !state.active {
        eprintln!("⚠️ STOP called but tracker not active, ignoring");
        return;
    }

    if let Some(stdin) = state.stdin.as_mut() {
        writeln!(stdin, "STOP").expect("failed to write STOP to the tracker");
        stdin.flush().unwrap();
        state.active = false;
    }
    if let Some(reader) = state.stdout.as_mut() {
        let mut ack = String::new();
        reader.read_line(&mut ack).expect("tracker bridge died on STOP");
        println!("[Rust] {}", ack.trim());
    }
}


/// Record per-epoch model quality.
///
/// Every ecosystem computed accuracy in the first campaign and every one only
/// printed it, so energy could never be normalised by the useful work produced.
pub fn log_metric(epoch: u32, train_loss: f64, test_loss: f64, test_acc: f64) {
    let mut state = TRACKER.lock().unwrap();
    if let Some(stdin) = state.stdin.as_mut() {
        let _ = writeln!(
            stdin,
            "METRIC epoch={} train_loss={:.6} test_loss={:.6} test_acc={:.4}",
            epoch, train_loss, test_loss, test_acc
        );
        let _ = stdin.flush();
    }
    if let Some(reader) = state.stdout.as_mut() {
        let mut ack = String::new();
        let _ = reader.read_line(&mut ack);
    }
}

/// Record what this stack's loader actually produced.
///
/// Seven ecosystems resize images with four different implementations, and only
/// three of them can be inspected from Python. Resizing is a no-op on CIFAR-100,
/// an upsample on Fashion-MNIST and a 2x downsample on Tiny ImageNet, where it
/// matters most: tf.image.resize defaults to no antialiasing and gave pixels
/// 3.8% wider in standard deviation than torchvision until that was corrected.
/// So each run records its own, and the campaign proves its own data parity.
pub fn log_data_fingerprint(n: i64, mean: f64, sd: f64, min: f64, max: f64) {
    let mut state = TRACKER.lock().unwrap();
    if let Some(stdin) = state.stdin.as_mut() {
        let _ = writeln!(
            stdin,
            "DATAFP split=test n={} mean={:.6} sd={:.6} min={:.6} max={:.6}",
            n, mean, sd, min, max
        );
        let _ = stdin.flush();
    }
    if let Some(reader) = state.stdout.as_mut() {
        let mut ack = String::new();
        let _ = reader.read_line(&mut ack);
    }
}

/// Repetition index, seed and epoch count for this run, from the shared contract.
pub fn run_params() -> (u32, u64, u32) {
    let get = |k: &str, d: u64| -> u64 {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let rep = get("DEEPGREEN_REP", 0) as u32;
    let seed = get("DEEPGREEN_SEED", 1000 + rep as u64);
    let epochs = get("DEEPGREEN_EPOCHS", 30) as u32;
    (rep, seed, epochs)
}

pub fn shutdown_tracker_daemon() {
    let mut state = TRACKER.lock().unwrap();

    if let Some(stdin) = state.stdin.as_mut() {
        let _ = writeln!(stdin, "EXIT");
        let _ = stdin.flush();   // 🔑 flush finale
        println!("[Rust] Sent EXIT to daemon");
    }

    if let Some(mut child) = state.child.take() {
        let _ = child.wait();
        println!("[Rust] Tracker daemon shut down");
    }

    state.stdin = None;
    state.stdout = None;
    state.active = false;
}