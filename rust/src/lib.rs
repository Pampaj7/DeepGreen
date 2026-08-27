pub mod datasets;
pub mod models;
pub mod emissions;

/// Resolve a dataset path.
///
/// The first campaign hard-coded `/home/pampaj/DeepGreen/data/...` in every
/// binary, so the replication package only ran on one machine. The root is now
/// taken from `DEEPGREEN_DATA` and falls back to `data/` relative to the
/// working directory.
pub fn data_path(relative: &str) -> String {
    let root = std::env::var("DEEPGREEN_DATA").unwrap_or_else(|_| "data".to_string());
    format!("{}/{}", root.trim_end_matches('/'), relative)
}

/// Cap the rayon pool used for image decoding.
///
/// The first campaign let rayon use every core (96 on the measurement server)
/// while PyTorch, C++ and Java used 2 loader workers, R used 0 and the Keras
/// generators used 1. Since the GPU runs well below its power limit in this
/// workload, loader parallelism dominates epoch duration and therefore energy,
/// so it must be held constant across ecosystems rather than left to each
/// framework's default. Override with `DEEPGREEN_LOADER_THREADS`.
pub fn init_loader_pool() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let n: usize = std::env::var("DEEPGREEN_LOADER_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    });
}

/// Whether to apply per-channel mean/std normalisation.
///
/// Only the Rust stack normalised in the first campaign; the other seven fed
/// raw [0,1] inputs. Off by default so the eight stacks solve the same
/// optimisation problem. Set `DEEPGREEN_NORMALIZE=1` to restore the old
/// behaviour.
pub fn normalize_inputs() -> bool {
    std::env::var("DEEPGREEN_NORMALIZE").map(|v| v == "1").unwrap_or(false)
}


/// Resolve the shared TorchScript module for one (architecture, dataset) pair.
///
/// Spec S1: the four LibTorch-based ecosystems must train the *same* module.
/// In the first campaign this stack built its own ResNet-18 and VGG-16 in
/// `src/models/`, so "C++ vs Rust vs Python vs R over one backend" actually
/// compared four different implementations.
///
/// The modules are produced by `scripts/export_torchscript_models.py` (and, for
/// the C++ build, by `cmake_script/GenerateModel.cmake`). The exporting torch
/// build must match the LibTorch this crate links against: a module written by a
/// newer torch will not load into an older LibTorch.
pub fn model_path(arch: &str, dataset: &str) -> String {
    let root = std::env::var("DEEPGREEN_MODELS").unwrap_or_else(|_| "models".to_string());
    format!("{}/{}_{}.pt", root.trim_end_matches('/'), arch, dataset)
}
