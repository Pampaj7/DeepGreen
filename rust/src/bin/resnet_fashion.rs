use rust::datasets::fashion::Fashion;
use rust::models::resnet::resnet18;
use rust::emissions::{init_tracker_daemon, start_tracker, stop_tracker, shutdown_tracker_daemon};

use tch::{nn, nn::OptimizerConfig, Device, Kind};
use tch::nn::ModuleT;
use std::collections::HashMap;
use rand::thread_rng;

fn main() {
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    init_tracker_daemon();

    // --- Load datasets
    let mut train_data = Fashion::new(
        "/home/pampaj/DeepGreen/data/fashion_mnist_png/train",
        device,
        None,
    ).unwrap();

    let test_data = Fashion::new(
        "/home/pampaj/DeepGreen/data/fashion_mnist_png/test",
        device,
        None,
    ).unwrap();

    println!("Train dataset size: {}", train_data.len());
    println!("Test dataset size: {}", test_data.len());

    let mut rng = thread_rng();

    // --- Model
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = resnet18(&root, 10); // 10 classi Fashion-MNIST

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        train_data.shuffle(&mut rng);

        // === Training
        let train_file = format!("resnet_fashion_train_epoch{:02}.csv", epoch);
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

        println!("Epoch {epoch}, avg train loss: {:.4}", total_loss / steps as f64);
                stop_tracker();

        // === Eval (item-per-item)
        let eval_file = format!("resnet_fashion_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            for batch in test_data.iter_batches(1) {
                let (x, y) = batch.unwrap(); // x: [1, 1, 28, 28]

                // Rimuovo batch → [1, 28, 28]
                let img = x.squeeze_dim(0);

                // Resize a 32x32
                let mut img_resized = tch::vision::image::resize(&img.to(Device::Cpu), 32, 32).unwrap();

                // Se è in scala di grigi, replico i canali → [3, 32, 32]
                if img_resized.size()[0] == 1 {
                    img_resized = img_resized.repeat(&[3, 1, 1]);
                }

                // Conversione a float e normalizzazione
                img_resized = img_resized.to_kind(Kind::Float) / 255.0;

                // Aggiungo batch dimension e porto su GPU
                let img_resized = img_resized.unsqueeze(0).to_device(device);

                // Forward
                let output = net.forward_t(&img_resized, false);

                let predicted = output
                    .argmax(-1, false)
                    .to(Device::Cpu)
                    .int64_value(&[]);

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
            println!("⚠️ WARNING: possible class collapse: {:?}", pred_class_hist);
        }

        stop_tracker();

    }

    shutdown_tracker_daemon();
    vs.save("resnet_fashion.ot").unwrap();
}
