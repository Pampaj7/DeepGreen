use rust::datasets::imagenette::Imagenette;
use tch::{nn, nn::OptimizerConfig, Device};
use tch::nn::ModuleT;
use std::collections::HashMap;
use std::time::Instant;
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
    // resize_to = None: dataloader/download_convert_imagenette.py already wrote
    // every image at 224x224 (shorter side to 224, centre crop), so there is
    // nothing to resize at run time -- spec S3, and the reason the saturation
    // cell can claim the eight stacks decode identical pixels.
    let mut train_data = Imagenette::new(
        &rust::data_path("imagenette_png/train"),
        device,
        None,
    ).unwrap();
    let test_data = Imagenette::new(
        &rust::data_path("imagenette_png/test"),
        device,
        None,
    ).unwrap();

    println!("Train dataset size: {}", train_data.len());
    println!("Test dataset size: {}", test_data.len());
    // The label mapping this run actually used, printed so it can be compared
    // with the other stacks' without re-deriving it from a directory listing.
    println!("Classes ({}): {}", train_data.classes().len(), train_data.classes().join(", "));
    assert_eq!(train_data.classes(), test_data.classes(),
               "train and test splits disagree on the class list");

    // --- Modello
    // Spec S1: load the shared TorchScript module rather than this crate's own
    // port, so that Rust/tch, C++/LibTorch, Python/PyTorch and R/torch all train
    // the identical torchvision graph. Parameters register into the VarStore, so
    // the optimizer must be built after the load.
    let vs = nn::VarStore::new(device);
    let mut net = tch::TrainableCModule::load(
        rust::model_path("vgg16", "imagenette"),
        vs.root(),
    )
    .expect("shared TorchScript module not found; run scripts/export_torchscript_models.py");
    net.set_train();

    // S1: the network this run trains, checked against the count the driver
    // carries out of models/MANIFEST.json. VGG-16 ran as four different
    // networks across the seven stacks under a specification that claimed the
    // counts were checked; nothing checked them. Empty when the manifest is
    // missing, which the driver allows and this stack treats as "unverified"
    // rather than as agreement.
    let n_params: i64 = vs.trainable_variables().iter().map(|t| t.numel() as i64).sum();
    match std::env::var("DEEPGREEN_EXPECTED_PARAMS").ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
    {
        Some(expected) => {
            assert_eq!(
                n_params, expected,
                "parameter count mismatch: the module has {} trainable parameters, \
                 the campaign expects {}", n_params, expected
            );
            println!("[Rust] parameters: {} (matches DEEPGREEN_EXPECTED_PARAMS)", n_params);
        }
        None => println!("[Rust] parameters: {} (DEEPGREEN_EXPECTED_PARAMS unset)", n_params),
    }

    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    // Saturation cell: 32, not the campaign's 128. VGG-16 at 224x224x3 does not
    // fit 24 GB at batch 128, and the two models must see the same batch size
    // for the cell to compare them.
    let batch_size = 32;

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

    // Seeded once, outside the epoch loop. Re-seeding inside it rebuilds the
    // same generator every epoch, so the network sees one permutation thirty
    // times instead of thirty.
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
        let train_start = Instant::now();

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

        let train_secs = train_start.elapsed().as_secs_f64();
        println!(
            "Epoch {epoch}, avg train loss: {:.4} ({} steps in {:.2}s)",
            total_loss / steps as f64,
            steps,
            train_secs
        );
        stop_tracker();

        // === Eval
        net.set_eval();
        start_tracker("eval", epoch);

        let mut correct: i64 = 0;
        let mut test_loss_sum = 0.0f64;
        let mut test_steps = 0i64;
        let mut pred_class_hist = HashMap::new();
        let eval_start = Instant::now();

        tch::no_grad(|| {
            // Batched evaluation, at the same batch size as training.
            // The first campaign evaluated one image at a time in every Rust
            // binary while all seven other ecosystems evaluated in batches;
            // batch-1 GPU inference is launch-overhead bound.
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

        let eval_secs = eval_start.elapsed().as_secs_f64();
        let acc = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}% (eval in {:.2}s)", acc, eval_secs);

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
