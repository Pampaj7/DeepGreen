use rust::datasets::tiny::TinyImageNet;
use rust::models::vgg::vgg16;
use rust::emissions::{init_tracker_daemon, start_tracker, stop_tracker, shutdown_tracker_daemon};

use std::collections::HashMap;
use tch::{nn, nn::OptimizerConfig, Device};
use tch::nn::ModuleT;
use rand::thread_rng;


fn main() {
    std::env::set_var("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128");

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    init_tracker_daemon();

    // --- Load datasets
    let mut train_data = TinyImageNet::new(
        "/home/pampaj/DeepGreen/data/tiny_imagenet_png/train",
        device,
        Some(32), // resize a 32×32 dentro al loader
    ).unwrap();

    let test_data = TinyImageNet::new(
        "/home/pampaj/DeepGreen/data/tiny_imagenet_png/val",
        device,
        Some(32),
    ).unwrap();

    println!("Train dataset size: {}", train_data.len());
    println!("Test dataset size: {}", test_data.len());

    let mut rng = thread_rng();

    // --- Model
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = vgg16(&root, 200); // 200 classi TinyImageNet

    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    let batch_size = 128;
    let epochs = 1;

    for epoch in 1..=epochs {
        train_data.shuffle(&mut rng);

        // === Training
        let train_file = format!("vgg_tinyimagenet_train_epoch{:02}.csv", epoch);
        start_tracker("emissions", &train_file);

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

        // === Evaluation
        let eval_file = format!("vgg_tinyimagenet_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            for batch in test_data.iter_batches(1) {
                let (x, y) = batch.unwrap();
                let output = net.forward_t(&x, false);

                let predicted = output.argmax(-1, false).int64_value(&[]);

                *pred_class_hist.entry(predicted).or_insert(0) += 1;

                if predicted == y.int64_value(&[]) {
                    correct += 1;
                }

                drop(output);
            }
        });

        let acc = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", acc);

        if pred_class_hist.len() <= 3 {
            println!(
                "⚠️ WARNING: Predicted classes are very few → possible class collapse: {:?}",
                pred_class_hist
            );
        }

        stop_tracker();
    }

    shutdown_tracker_daemon();
}
