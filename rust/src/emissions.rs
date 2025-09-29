use std::process::{Command, Child, ChildStdin, Stdio};
use std::io::Write;
use std::path::PathBuf;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::{fs, thread, time::Duration};

struct TrackerState {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    active: bool,
}

static TRACKER: Lazy<Mutex<TrackerState>> = Lazy::new(|| {
    Mutex::new(TrackerState {
        child: None,
        stdin: None,
        active: false,
    })
});

fn get_daemon_script() -> PathBuf {
    let base = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(base).join("scripts").join("tracker_daemon.py")
}

pub fn init_tracker_daemon() {
    let script_path = get_daemon_script();
    println!("[Rust] Launching tracker daemon: {:?}", script_path);

    let mut child = Command::new("python3")
        .arg(script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("❌ Failed to start tracker daemon");

    let stdin = child.stdin.take().expect("❌ Failed to get stdin for tracker daemon");

    let mut state = TRACKER.lock().unwrap();
    state.child = Some(child);
    state.stdin = Some(stdin);
    state.active = false;

    println!(
        "[Rust] Tracker daemon started (PID: {:?})",
        state.child.as_ref().unwrap().id()
    );
}

pub fn start_tracker(output_dir: &str, output_file: &str) {
    // fix per lock di CodeCarbon
    let _ = fs::remove_file("/tmp/.codecarbon.lock");

    let mut state = TRACKER.lock().unwrap();
    if state.active {
        eprintln!("⚠️ Tracker already active, ignoring START for {}", output_file);
        return;
    }

    if let Some(stdin) = state.stdin.as_mut() {
        writeln!(stdin, "START {} {}", output_dir, output_file)
            .expect("❌ Failed to write START to tracker stdin");
        stdin.flush().unwrap();   // 🔑 forza flush immediato
        println!("[Rust] Sent START for {}", output_file);
        state.active = true;
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
        writeln!(stdin, "STOP").expect("❌ Failed to write STOP to tracker");
        stdin.flush().unwrap();
        println!("[Rust] Sent STOP");
        state.active = false;
    }
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
    state.active = false;
}