use rust::datasets::fashion::Fashion;
use rust::models::vgg::vgg16;
use rust::emissions::{init_tracker_daemon, start_tracker, stop_tracker, shutdown_tracker_daemon};

use tch::{nn, nn::OptimizerConfig, Device, Kind};
use tch::nn::ModuleT;
use tch::vision::image::resize;
use std::collections::HashMap;
use rand::thread_rng;

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
    let net = vgg16(&root, 10); // 10 classi Fashion-MNIST

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        train_data.shuffle(&mut rng);

        // === Training
        let train_file = format!("vgg_fashion_train_epoch{:02}.csv", epoch);
        start_tracker("emissions", &train_file);

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
        let eval_file = format!("vgg_fashion_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            for batch in test_data.iter_batches(1) {
                let (x, y) = batch.unwrap();

                let x = preprocess_batch(&x, device);

                let output = net.forward_t(&x, false);
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
    vs.save("vgg_fashion.ot").unwrap();
}
