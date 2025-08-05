use rust::datasets::cifar100::load_cifar100;
use rust::models::resnet::resnet18;
use tch::{nn, nn::OptimizerConfig, Tensor, Device, Kind};
use tch::nn::ModuleT;
use std::collections::HashMap;
use rand::seq::SliceRandom; // in main.rs
use rust::emissions::{start_tracker, stop_tracker};
use rust::emissions::{init_tracker_daemon, shutdown_tracker_daemon}; // aggiungi anche questi due

fn main() {
    init_tracker_daemon();
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // --- Load datasets
    let mut train_data = load_cifar100("/home/pampaj/DeepGreen/data/cifar100_png/train", device, None).unwrap();
    let test_data = load_cifar100("/home/pampaj/DeepGreen/data/cifar100_png/test", device, None).unwrap();
    
    let mut rng = rand::thread_rng();
    train_data.shuffle(&mut rng);
    println!("Train size: {}, Test size: {}", train_data.len(), test_data.len());

    // --- Model
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = resnet18(&root, 100);

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        // --- Training

        // Start emissions tracker
        let train_file = format!("resnet_cifar100_train_epoch{:02}.csv", epoch);
        start_tracker("emissions", &train_file);

        let mut total_loss = 0.0;
        for (batch_idx, batch) in train_data.chunks(batch_size).enumerate() {
            let images: Vec<_> = batch.iter().map(|(img, _)| img.unsqueeze(0)).collect();
            let labels: Vec<_> = batch.iter().map(|(_, label)| *label).collect();

            let input = Tensor::cat(&images, 0);
            let target = Tensor::from_slice(&labels)
                .to_kind(Kind::Int64)
                .to_device(device);

            if batch_idx < 3 {
                let mut label_count = HashMap::new();
                for l in &labels {
                    *label_count.entry(*l).or_insert(0) += 1;
                }
            }

            let output = net.forward_t(&input, true);

            if output.size()[1] != 100 {
                println!("⚠️ WARNING: Output dimension is {:?} instead of [B, 100]", output.size());
            }

            let loss = output.cross_entropy_for_logits(&target);
            opt.backward_step(&loss);

            total_loss += loss.double_value(&[]);
        }

        let avg_loss = total_loss / (train_data.len() as f64 / batch_size as f64);
        println!("Epoch {epoch}, avg train loss: {:.4}", avg_loss);

        // -- STOP tracker: training
        stop_tracker();


        // -- START tracker: evaluation
        let eval_file = format!("resnet_cifar100_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        // --- Test
        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        for (i, (img, label)) in test_data.iter().enumerate() {
            let output = net.forward_t(&img.unsqueeze(0), false);
            let predicted = output.argmax(-1, false).int64_value(&[]);

            *pred_class_hist.entry(predicted).or_insert(0) += 1;


            if predicted == *label {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", accuracy);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: Predicted classes are very few → possible class collapse: {:?}", pred_class_hist);
        }

        // -- STOP tracker: evaluation
        stop_tracker();
    }
    shutdown_tracker_daemon();
}


