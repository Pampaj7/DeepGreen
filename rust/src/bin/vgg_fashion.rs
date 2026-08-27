use rust::datasets::fashion::Fashion;
use rust::emissions::{init_tracker_daemon, start_tracker, stop_tracker, shutdown_tracker_daemon};

use tch::{nn, nn::OptimizerConfig, Device, Kind};
use tch::nn::ModuleT;
use tch::vision::image::resize;
use std::collections::HashMap;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Converte grayscale [1,H,W] in RGB [3,H,W]
fn expand_to_rgb(img: &tch::Tensor) -> tch::Tensor {
    if img.size()[0] == 1 {
        img.repeat(&[3, 1, 1])
    } else {
        img.shallow_clone()
    }
}

/// Prepara immagine singola: resize a 32×32 + RGB + float normalizzato
fn preprocess(img: &tch::Tensor, device: Device) -> tch::Tensor {
    let mut x_resized = resize(&img.to(Device::Cpu), 32, 32).unwrap();
    x_resized = expand_to_rgb(&x_resized);
    x_resized = x_resized.to_kind(Kind::Float) / 255.0;
    x_resized.to_device(device)
}

/// Preprocessa un batch intero [B,1,H,W] → [B,3,32,32]
fn preprocess_batch(x: &tch::Tensor, device: Device) -> tch::Tensor {
    let mut images: Vec<tch::Tensor> = Vec::new();
    for i in 0..x.size()[0] {
        let img = x.get(i);
        let img = preprocess(&img, device);
        images.push(img.unsqueeze(0));
    }
    tch::Tensor::cat(&images, 0)
}

fn main() {
    let device = Device::cuda_if_available();
    // Repetition, seed and epoch count come from the shared run contract
    // (tools/deepgreen_tracker.py); the first campaign hard-coded 30 epochs
    // and had no notion of an independent repetition at all.
    let (rep, seed, epochs) = rust::emissions::run_params();
    println!("[Rust] repetition {} seed {} epochs {}", rep, seed, epochs);
    println!("Using device: {:?}", device);

    init_tracker_daemon();

    // --- Load datasets
    let mut train_data = Fashion::new(
        &rust::data_path("fashion_mnist_png/train"),
        device,
        None,
    ).unwrap();

    let test_data = Fashion::new(
        &rust::data_path("fashion_mnist_png/test"),
        device,
        None,
    ).unwrap();

    println!("Train dataset size: {}", train_data.len());
    println!("Test dataset size: {}", test_data.len());

    let mut rng = StdRng::seed_from_u64(seed);

    // --- Model
    let vs = nn::VarStore::new(device);
    // Spec S1: load the shared TorchScript module rather than this crate's own
    // port, so that Rust/tch, C++/LibTorch, Python/PyTorch and R/torch all train
    // the identical torchvision graph. Parameters register into the VarStore, so
    // the optimizer must be built after the load.
    let mut net = tch::TrainableCModule::load(
        rust::model_path("vgg16", "fashionmnist"),
        vs.root(),
    )
    .expect("shared TorchScript module not found; run scripts/export_torchscript_models.py");
    net.set_train();

    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    let batch_size = 128;

    for epoch in 1..=epochs {
        train_data.shuffle(&mut rng);

        // === Training
        // TrainableCModule ignores the bool in forward_t: the mode is module
        // state, set here. Without this the evaluation runs with batch norm in
        // training mode -- the same defect found in the TensorFlow stack.
        net.set_train();
        start_tracker("train", epoch);

        let mut total_loss = 0.0;
        let mut steps = 0;

        for batch in train_data.iter_batches(batch_size) {
            let (x, y) = batch.unwrap();

            // preprocess intero batch
            let x = preprocess_batch(&x, device);

            let output = net.forward_t(&x, true);
            let loss = output.cross_entropy_for_logits(&y);
            opt.backward_step(&loss);

            total_loss += loss.double_value(&[]);
            steps += 1;

            drop(output);
            drop(loss);
        }

        println!("Epoch {epoch}, avg train loss: {:.4}", total_loss / steps as f64);
        stop_tracker();

        // === Eval (item-per-item)
        net.set_eval();
        start_tracker("eval", epoch);

        let mut correct: i64 = 0;
        let mut test_loss_sum = 0.0f64;
        let mut test_steps = 0i64;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            // Batched evaluation, at the same batch size as training.
            // The first campaign evaluated one image at a time
            // (test_data.iter_batches(1)) in every Rust binary, while all seven
            // other ecosystems evaluated at 128. Batch-1 GPU inference is
            // launch-overhead bound, which inflated this stack's inference cost
            // and is the most likely source of its train/inference ranking
            // reversal -- a result the manuscript reports as a finding.
            for batch in test_data.iter_batches(batch_size) {
                let (x, y) = batch.unwrap();
                let output = net.forward_t(&x, false);
                test_loss_sum += output.cross_entropy_for_logits(&y).double_value(&[]);
                test_steps += 1;
                let predicted = output.argmax(-1, false);

                correct += predicted
                    .eq_tensor(&y)
                    .sum(tch::Kind::Int64)
                    .int64_value(&[]);

                let preds_cpu = predicted.to(Device::Cpu);
                for i in 0..preds_cpu.size()[0] {
                    *pred_class_hist.entry(preds_cpu.int64_value(&[i])).or_insert(0) += 1;
                }

                drop(output);
            }
        });

        let acc = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", acc);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: possible class collapse: {:?}", pred_class_hist);
        }

        stop_tracker();

        // Outside the tracked block: writing the metric must not be measured.
        let test_loss = if test_steps > 0 { test_loss_sum / test_steps as f64 } else { f64::NAN };
        rust::emissions::log_metric(epoch, total_loss / steps as f64, test_loss, acc);
    }

    shutdown_tracker_daemon();
    vs.save("vgg_fashion.ot").unwrap();
}
