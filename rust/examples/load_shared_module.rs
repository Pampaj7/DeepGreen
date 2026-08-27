//! Smoke test for spec S1: the shared TorchScript module must load into this
//! stack and produce the expected output shape.
//!
//! Run after `python3 scripts/export_torchscript_models.py`:
//!
//!     LIBTORCH=/path/to/libtorch cargo run --example load_shared_module
//!
//! A module exported by a torch newer than the LibTorch this crate links
//! against will fail here rather than at hour three of a campaign.

use tch::nn::ModuleT;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() {
    let device = Device::cuda_if_available();
    println!("device: {:?}", device);

    let cases = [
        ("resnet18", "fashionmnist", 10i64),
        ("resnet18", "cifar100", 100),
        ("resnet18", "tinyimagenet200", 200),
        ("vgg16", "fashionmnist", 10),
        ("vgg16", "cifar100", 100),
        ("vgg16", "tinyimagenet200", 200),
    ];

    let mut failures = 0;
    for (arch, dataset, num_classes) in cases {
        let path = rust::model_path(arch, dataset);
        let vs = nn::VarStore::new(device);
        match tch::TrainableCModule::load(&path, vs.root()) {
            Ok(mut net) => {
                net.set_train();
                let x = Tensor::zeros([2, 3, 32, 32], (Kind::Float, device));
                let out = net.forward_t(&x, true);
                let shape = out.size();
                let params: i64 = vs.trainable_variables().iter().map(|t| t.numel() as i64).sum();
                let ok = shape == vec![2, num_classes];
                if !ok {
                    failures += 1;
                }
                println!(
                    "  {:<8} {:<16} {:>12} params  out {:?}  {}",
                    arch, dataset, params, shape,
                    if ok { "ok" } else { "SHAPE MISMATCH" }
                );

                // the module must also be trainable through the VarStore
                if let Ok(mut opt) = nn::Adam::default().build(&vs, 1e-4) {
                    let y = Tensor::zeros([2], (Kind::Int64, device));
                    let loss = net.forward_t(&x, true).cross_entropy_for_logits(&y);
                    opt.backward_step(&loss);
                } else {
                    println!("      optimizer could not be built over the loaded module");
                    failures += 1;
                }
            }
            Err(e) => {
                failures += 1;
                println!("  {:<8} {:<16} FAILED to load {}: {}", arch, dataset, path, e);
            }
        }
    }

    if failures > 0 {
        eprintln!("\n{} module(s) failed; check models/MANIFEST.txt against the tch/LibTorch version", failures);
        std::process::exit(1);
    }
    println!("\nall 6 shared modules load, forward and train through the VarStore");
}
