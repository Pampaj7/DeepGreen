use rust::datasets::fashion::load_fashion_mnist;
use rust::models::vgg::{vgg16_tiny, init_weights};
use rust::emissions::{init_tracker_daemon, start_tracker, stop_tracker, shutdown_tracker_daemon};

use tch::{nn, nn::OptimizerConfig, Tensor, Device, Kind};
use tch::nn::ModuleT;
use tch::vision::image::resize;
use std::collections::HashMap;
use rand::seq::SliceRandom;

fn main() {
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    init_tracker_daemon();

    // --- Load datasets
    let mut train_data = load_fashion_mnist("/home/pampaj/DeepGreen/data/fashion_mnist_png/train", device).unwrap();
    let mut test_data = load_fashion_mnist("/home/pampaj/DeepGreen/data/fashion_mnist_png/test", device).unwrap();

    // --- Resize to 32×32
    fn resize_dataset(data: &mut Vec<(Tensor, i64)>, new_size: i64, device: Device) {
        for (img, _) in data.iter_mut() {
            let img_cpu = img.to_device(Device::Cpu);
            let resized = resize(&img_cpu, new_size, new_size).unwrap();
            let img_resized = resized.to_kind(Kind::Float).to_device(device) / 255.0;
            *img = img_resized;
        }
    }

    resize_dataset(&mut train_data, 32, device);
    resize_dataset(&mut test_data, 32, device);

    // Shuffle training data
    let mut rng = rand::thread_rng();
    train_data.shuffle(&mut rng);
    println!("Train size: {}, Test size: {}", train_data.len(), test_data.len());

    // --- Model
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = vgg16_tiny(&root, 10);
    // init_weights(&vs); // opzionale

    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        // --- Start tracker: TRAIN
        let train_file = format!("vgg_fashion_train_epoch{:02}.csv", epoch);
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
                tch::no_grad(|| {
                    println!("[Debug Train] Batch {batch_idx}: input shape = {:?}", input.size());
                    let mut label_count = HashMap::new();
                    for l in &labels {
                        *label_count.entry(*l).or_insert(0) += 1;
                    }
                    println!("[Debug Train] Label distribution: {:?}", label_count);
                });
            }

            let output = net.forward_t(&input, true);
            if output.size()[1] != 10 {
                println!("⚠️ WARNING: Output dimension is {:?} instead of [B, 10]", output.size());
            }

            let loss = output.cross_entropy_for_logits(&target);
            opt.backward_step(&loss);
            total_loss += loss.double_value(&[]);

            drop(input);
            drop(target);
            drop(output);
        }

        let avg_loss = total_loss / (train_data.len() as f64 / batch_size as f64);
        println!("Epoch {epoch}, avg train loss: {:.4}", avg_loss);

        // --- Stop tracker: TRAIN
        stop_tracker();

        // --- Start tracker: EVAL
        let eval_file = format!("vgg_fashion_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        // --- Test
        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            for (i, (img, label)) in test_data.iter().enumerate() {
                let output = net.forward_t(&img.unsqueeze(0), false);
                let predicted = output
                    .argmax(-1, false)
                    .to(Device::Cpu)
                    .int64_value(&[]);

                *pred_class_hist.entry(predicted).or_insert(0) += 1;

                if predicted == *label {
                    correct += 1;
                }

                drop(output);
            }
        });

        let accuracy = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", accuracy);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: Predicted classes are very few → possible class collapse: {:?}", pred_class_hist);
        }

        // --- Stop tracker: EVAL
        stop_tracker();
    }

    shutdown_tracker_daemon();
    vs.save("vgg_fashion_32x32.ot").unwrap();
}
