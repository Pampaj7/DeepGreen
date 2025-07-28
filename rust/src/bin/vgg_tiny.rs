use rust::datasets::tiny::load_tiny_imagenet;
use rust::models::vgg::vgg16;
use rust::models::vgg::vgg16_tiny;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use tch::{nn, nn::OptimizerConfig, Tensor, Device, Kind, vision::image::resize};
use tch::nn::ModuleT;

fn main() {
    std::env::set_var("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128");

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let mut train_data = load_tiny_imagenet("/home/pampaj/DeepGreen/data/tiny_imagenet_png/train", Device::Cpu).unwrap();
    let mut test_data = load_tiny_imagenet("/home/pampaj/DeepGreen/data/tiny_imagenet_png/val", Device::Cpu).unwrap();
    let mut rng = rand::thread_rng();
    train_data.shuffle(&mut rng);
    println!("Train size: {}, Test size: {}", train_data.len(), test_data.len());

    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = vgg16_tiny(&root, 200);

    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    let batch_size = 64;
    let epochs = 10;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;

        for (_batch_idx, batch) in train_data.chunks(batch_size).enumerate() {
            let images: Vec<_> = batch.iter().map(|(img, _)| img.unsqueeze(0).to_device(device)).collect();
            let labels: Vec<_> = batch.iter().map(|(_, label)| *label).collect();

            let input = Tensor::cat(&images, 0);
            let target = Tensor::from_slice(&labels).to_kind(Kind::Int64).to_device(device);

            let output = net.forward_t(&input, true);
            if output.size()[1] != 200 {
                println!("⚠️ WARNING: Output dim is {:?} instead of [B, 200]", output.size());
            }

            let loss = output.cross_entropy_for_logits(&target);
            opt.backward_step(&loss);
            total_loss += loss.double_value(&[]);

            drop(input);
            drop(target);
            drop(output);
            drop(loss);
        }

        let avg_loss = total_loss / (train_data.len() as f64 / batch_size as f64);
        println!("Epoch {epoch}, avg train loss: {:.4}", avg_loss);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();

        tch::no_grad(|| {
            for (_i, (img, label)) in test_data.iter().enumerate() {
                let img_input = img.unsqueeze(0).to_device(device);
                let output = net.forward_t(&img_input, false);
                let predicted = output.argmax(-1, false).int64_value(&[]);

                *pred_class_hist.entry(predicted).or_insert(0) += 1;

                if predicted == *label {
                    correct += 1;
                }

                drop(img_input);
                drop(output);
            }
        });

        let accuracy = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", accuracy);

        if pred_class_hist.len() <= 3 {
            println!("⚠️ WARNING: Predicted classes are very few → possible class collapse: {:?}", pred_class_hist);
        }
    }

    vs.save("vgg_tinyimagenet.ot").unwrap();
}
