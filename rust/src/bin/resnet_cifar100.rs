use rust::datasets::cifar100::Cifar100;
use rust::models::resnet::resnet18;
use tch::{nn, nn::OptimizerConfig, Device, Kind};
use tch::nn::ModuleT;
use std::collections::HashMap;
use rand::thread_rng;
use rust::emissions::{init_tracker_daemon, shutdown_tracker_daemon, start_tracker, stop_tracker};

fn main() {
    init_tracker_daemon();
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // --- Dataset stile PyTorch
    let mut train_data = Cifar100::new("/home/pampaj/DeepGreen/data/cifar100_png/train", device, None).unwrap();
    let test_data = Cifar100::new("/home/pampaj/DeepGreen/data/cifar100_png/test", device, None).unwrap();

    // --- Modello
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = resnet18(&root, 100);
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        let mut rng = thread_rng();
        train_data.shuffle(&mut rng);

        // === Training
        let train_file = format!("resnet_cifar100_train_epoch{:02}.csv", epoch);
        start_tracker("train", &train_file);

        let mut total_loss = 0.0;
        let mut steps = 0;

        for batch in train_data.iter_batches(batch_size) {
            let (x, y) = batch.unwrap();
            let output = net.forward_t(&x, true);
            let loss = output.cross_entropy_for_logits(&y);
            opt.backward_step(&loss);

            total_loss += f64::from(&loss);
            steps += 1;
        }

        println!("Epoch {epoch}, avg train loss: {:.4}", total_loss / steps as f64);
        stop_tracker();

        // === Eval
        let eval_file = format!("resnet_cifar100_eval_epoch{:02}.csv", epoch);
        start_tracker("eval", &eval_file);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        for batch in test_data.iter_batches(batch_size) {
            let (x, y) = batch.unwrap();
            let output = net.forward_t(&x, false);
            let preds = output.argmax(-1, false);

            let eq = preds.eq1(&y).to_kind(Kind::Int64);
            correct += i64::from(eq.sum(Kind::Int64));

            for p in preds.into_iter::<i64>().unwrap() {
                *pred_class_hist.entry(p).or_insert(0) += 1;
            }
        }

        let acc = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", acc);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: possible class collapse: {:?}", pred_class_hist);
        }

        stop_tracker();
    }

    shutdown_tracker_daemon();
}
