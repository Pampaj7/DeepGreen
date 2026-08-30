use rust::datasets::cifar100::Cifar100;
use tch::{nn, nn::OptimizerConfig, Device, Kind};
use tch::nn::ModuleT;
use std::collections::HashMap;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rust::emissions::{init_tracker_daemon, shutdown_tracker_daemon, start_tracker, stop_tracker};

fn main() {
    init_tracker_daemon();
    let device = Device::cuda_if_available();
    // Repetition, seed and epoch count come from the shared run contract
    // (tools/deepgreen_tracker.py); the first campaign hard-coded 30 epochs
    // and had no notion of an independent repetition at all.
    let (rep, seed, epochs) = rust::emissions::run_params();
    println!("[Rust] repetition {} seed {} epochs {}", rep, seed, epochs);
    println!("Using device: {:?}", device);

    // --- Dataset stile PyTorch
    let mut train_data = Cifar100::new(
        &rust::data_path("cifar100_png/train"),
        device,
        None,
    ).unwrap();
    let test_data = Cifar100::new(
        &rust::data_path("cifar100_png/test"),
        device,
        None,
    ).unwrap();

    println!("Train dataset size: {}", train_data.len());
    println!("Test dataset size: {}", test_data.len());

    // --- Modello
    // Spec S1: load the shared TorchScript module rather than this crate's own
    // port, so that Rust/tch, C++/LibTorch, Python/PyTorch and R/torch all train
    // the identical torchvision graph. Parameters register into the VarStore, so
    // the optimizer must be built after the load.
    let vs = nn::VarStore::new(device);
    let mut net = tch::TrainableCModule::load(
        rust::model_path("resnet18", "cifar100"),
        vs.root(),
    )
    .expect("shared TorchScript module not found; run scripts/export_torchscript_models.py");
    net.set_train();
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    let batch_size = 128;

    // What this stack's loader actually produced, over the whole test split.
    // A batch is comparable across stacks only if it holds the same images, and
    // which images it holds depends on the order the loader enumerates files --
    // so a per-batch fingerprint measures enumeration order and pixel handling
    // together. Over every image it depends on the set, not the order.
    if std::env::var("DEEPGREEN_DATA_FINGERPRINT").as_deref() != Ok("0") {
        let (mut n, mut sum, mut sumsq) = (0i64, 0f64, 0f64);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for batch in test_data.iter_batches(batch_size) {
            let (x, _) = batch.unwrap();
            n += x.numel() as i64;
            sum += x.sum(tch::Kind::Double).double_value(&[]);
            sumsq += (&x * &x).sum(tch::Kind::Double).double_value(&[]);
            lo = lo.min(x.min().double_value(&[]));
            hi = hi.max(x.max().double_value(&[]));
        }
        if n > 0 {
            let mean = sum / n as f64;
            let sd = (sumsq / n as f64 - mean * mean).max(0.0).sqrt();
            rust::emissions::log_data_fingerprint(n, mean, sd, lo, hi);
        }
    }

    // Seeded once, outside the epoch loop. Re-seeding inside it -- which two of
    // these six binaries did -- rebuilds the same generator every epoch, so the
    // network sees one permutation thirty times instead of thirty. The other
    // four already did this correctly, which is exactly the kind of divergence
    // between siblings the conformance checker could not see while its parity
    // checks passed on any one file in a glob.
    let mut rng = StdRng::seed_from_u64(seed);

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
            let output = net.forward_t(&x, true);
            let loss = output.cross_entropy_for_logits(&y);
            opt.backward_step(&loss);

            total_loss += loss.double_value(&[]);
            steps += 1;

            drop(output);
            drop(loss);
        }

        println!(
            "Epoch {epoch}, avg train loss: {:.4}",
            total_loss / steps as f64
        );
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
}
