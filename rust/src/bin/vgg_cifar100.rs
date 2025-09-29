use rust::datasets::cifar100::Cifar100;
use rust::models::vgg::vgg16; // supponendo tu abbia un modulo vgg16
use tch::{nn, nn::OptimizerConfig, Device, Kind};
use tch::nn::ModuleT;
use std::collections::HashMap;
use rand::thread_rng;
use rust::emissions::{init_tracker_daemon, shutdown_tracker_daemon, start_tracker, stop_tracker};
use std::time::Instant;

fn main() {
    init_tracker_daemon();
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // --- Dataset stile PyTorch
    let mut train_data = Cifar100::new(
        "/home/pampaj/DeepGreen/data/cifar100_png/train",
        device,
        None,
    )
    .unwrap();
    let test_data = Cifar100::new(
        "/home/pampaj/DeepGreen/data/cifar100_png/test",
        device,
        None,
    )
    .unwrap();

    println!("Train dataset size: {}", train_data.len());
    println!("Test dataset size: {}", test_data.len());

    // --- Modello
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = vgg16(&root, 100);
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        // shuffle dataset ogni epoch
        let mut rng = thread_rng();
        train_data.shuffle(&mut rng);

        // === Training
        let train_file = format!("vgg_cifar100_train_epoch{:02}.csv", epoch);
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
        }

        println!(
            "Epoch {epoch}, avg train loss: {:.4}",
            total_loss / steps as f64
        );
        stop_tracker();

                // === Eval item-per-item
        let eval_file = format!("vgg_cifar100_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        let t0 = Instant::now();
        tch::no_grad(|| {
            for (i, batch) in test_data.iter_batches(1).enumerate() {
                let (img, label) = batch.unwrap();

                // img.shape = [1, 3, 32, 32], togliamo la dimensione batch
                let output = net.forward_t(&img, false);
                let predicted = output
                    .argmax(-1, false)
                    .to(Device::Cpu)
                    .int64_value(&[]);

                *pred_class_hist.entry(predicted).or_insert(0) += 1;

                if predicted == label.int64_value(&[]) {
                    correct += 1;
                }

                drop(output);
            }
        });


        let eval_time = t0.elapsed();
        println!("Epoch {epoch}, eval duration: {:?}", eval_time);

        let acc = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", acc);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: possible class collapse: {:?}", pred_class_hist);
        }

        stop_tracker();


        // aggiungiamo sleep per permettere a CodeCarbon di scrivere i dati

    }

    shutdown_tracker_daemon();
}
