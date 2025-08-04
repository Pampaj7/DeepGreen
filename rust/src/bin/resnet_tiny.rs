use rust::datasets::tiny::load_tiny_imagenet;
use rust::models::resnet::resnet18;
use tch::{nn, nn::OptimizerConfig, Tensor, Device, Kind, vision::image::resize};
use tch::nn::ModuleT;
use std::collections::HashMap;
use rand::seq::SliceRandom;

fn main() {
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let num_classes = 200;

    // --- Load datasets
    let mut train_data = load_tiny_imagenet("/home/pampaj/DeepGreen/data/tiny_imagenet_png/train", device).unwrap();
    let test_data = load_tiny_imagenet("/home/pampaj/DeepGreen/data/tiny_imagenet_png/val", device).unwrap();
    let mut rng = rand::thread_rng();
    train_data.shuffle(&mut rng);
    println!("Train size: {}, Test size: {}", train_data.len(), test_data.len());

    // --- Model
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = resnet18(&root, num_classes);

    let mut opt = nn::Adam {
        wd: 1e-4,
        ..Default::default()
    }.build(&vs, 1e-3).unwrap();
    let batch_size = 128;
    let epochs = 30;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;

        for (batch_idx, batch) in train_data.chunks(batch_size).enumerate() {
            let images: Vec<_> = batch.iter().map(|(img, _)| {
                let resized = resize(&img.to_device(Device::Cpu), 32, 32).unwrap();
                (resized / 255.0).to_device(device).unsqueeze(0)
            }).collect();

            let labels: Vec<_> = batch.iter().map(|(_, label)| *label).collect();

            let input = Tensor::cat(&images, 0);
            let target = Tensor::from_slice(&labels).to_kind(Kind::Int64).to_device(device);

            if batch_idx < 3 {
                println!("[Debug Train] Batch {batch_idx}: input shape = {:?}", input.size());
                let mut label_count = HashMap::new();
                for l in &labels {
                    *label_count.entry(*l).or_insert(0) += 1;
                }
                println!("[Debug Train] Label distribution: {:?}", label_count);
            }

            let output = net.forward_t(&input, true);

            if output.size()[1] != num_classes {
                println!("⚠️ WARNING: Output dimension is {:?} instead of [B, {}]", output.size(), num_classes);
            }

            let loss = output.cross_entropy_for_logits(&target);
            opt.backward_step(&loss);
            total_loss += loss.double_value(&[]);
        }

        let avg_loss = total_loss / (train_data.len() as f64 / batch_size as f64);
        println!("Epoch {epoch}, avg train loss: {:.4}", avg_loss);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            for (i, (img, label)) in test_data.iter().enumerate() {
                let resized = resize(&img.to_device(Device::Cpu), 32, 32).unwrap();
                let input = (resized / 255.0).to_device(device).unsqueeze(0);

                let output = net.forward_t(&input, false);
                let predicted = output.argmax(-1, false).int64_value(&[]);

                *pred_class_hist.entry(predicted).or_insert(0) += 1;

                if i < 10 {
                    println!("[Debug Test] Sample {i}: GT = {label}, Pred = {predicted}");
                }

                if predicted == *label {
                    correct += 1;
                }
            }
        });

        let accuracy = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", accuracy);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: Predicted classes are very few → possible class collapse: {:?}", pred_class_hist);
        }
    }

    vs.save("resnet_tinyimagenet.ot").unwrap();
}
