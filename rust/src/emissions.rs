use std::process::{Command, Child, ChildStdin, Stdio};
use std::io::Write;
use std::path::PathBuf;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::fs;

static TRACKER_PROCESS: Lazy<Mutex<Option<(Child, ChildStdin)>>> = Lazy::new(|| Mutex::new(None));

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

    println!("[Rust] Tracker daemon PID: {}", child.id()); // ✅ ORA child è definito

    let Some(stdin) = child.stdin.take() else {
        panic!("❌ Failed to get stdin for tracker daemon");
};

*TRACKER_PROCESS.lock().unwrap() = Some((child, stdin));
println!("[Rust] Tracker daemon started");

}

pub fn start_tracker(output_dir: &str, output_file: &str) {
    // fix per lock di codecarbon
    let _ = fs::remove_file("/tmp/.codecarbon.lock");

    if let Some((_, ref mut stdin)) = *TRACKER_PROCESS.lock().unwrap() {
        writeln!(stdin, "START {} {}", output_dir, output_file)
            .expect("Failed to write START to tracker stdin");
    } else {
        panic!("Tracker daemon not initialized");
    }
}


pub fn stop_tracker() {
    if let Some((_, ref mut stdin)) = *TRACKER_PROCESS.lock().unwrap() {
        writeln!(stdin, "STOP").expect("Failed to write STOP to tracker");
    }
}

pub fn shutdown_tracker_daemon() {
    let mut tracker_guard = TRACKER_PROCESS.lock().unwrap();
    if let Some((child, stdin)) = tracker_guard.as_mut() {
        let _ = writeln!(stdin, "EXIT");
        let _ = child.wait();
        println!("[Rust] Tracker daemon shut down");
    }
}
