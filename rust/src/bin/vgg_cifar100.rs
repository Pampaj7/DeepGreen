use rust::datasets::cifar100::load_cifar100;
use rust::models::vgg::vgg16;
use rust::emissions::{init_tracker_daemon, shutdown_tracker_daemon, start_tracker, stop_tracker};
use tch::{nn, nn::OptimizerConfig, nn::ModuleT, Device, Kind, Tensor, Cuda};
use std::collections::HashMap;
use rand::seq::SliceRandom;

fn main() {
    init_tracker_daemon();
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // --- Load datasets
    let mut train_data = load_cifar100(
        "/home/pampaj/DeepGreen/data/cifar100_png/train",
        device,
        None,
    )
    .unwrap();
    let mut test_data = load_cifar100(
        "/home/pampaj/DeepGreen/data/cifar100_png/test",
        device,
        None,
    )
    .unwrap();

    // Debug: dataset size
    println!(
        "DEBUG → Train size: {}, Test size: {}",
        train_data.len(),
        test_data.len()
    );

    // Normalizzazione [0,1]
    for (img, _) in train_data.iter_mut() {
        *img = img.f_div_scalar(255.0).unwrap();
    }
    for (img, _) in test_data.iter_mut() {
        *img = img.f_div_scalar(255.0).unwrap();
    }

    let mut rng = rand::thread_rng();

    // --- Model
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net = vgg16(&root, 100);

    // Adam con weight decay e learning rate più basso
    let mut opt = nn::Adam {
        wd: 5e-4,
        ..Default::default()
    }
    .build(&vs, 1e-4)
    .unwrap();

    let batch_size = 128;
    let epochs = 30; // puoi aumentare a 30 per esperimenti seri

    for epoch in 1..=epochs {
        // --- Training
        train_data.shuffle(&mut rng);

        let train_file = format!("vgg_cifar100_train_epoch{:02}.csv", epoch);
        start_tracker("emissions", &train_file);

        let mut total_loss = 0.0;
        let mut batches = 0;

        println!(
            "DEBUG → Epoch {epoch}: total train samples {}, batch_size {}",
            train_data.len(),
            batch_size
        );

        for batch in train_data.chunks(batch_size) {
            batches += 1;
            let images: Vec<_> = batch.iter().map(|(img, _)| img.unsqueeze(0)).collect();
            let labels: Vec<_> = batch.iter().map(|(_, label)| *label).collect();

            let input = Tensor::cat(&images, 0).to_device(device);
            let target = Tensor::from_slice(&labels)
                .to_kind(Kind::Int64)
                .to_device(device);

            let output = net.forward_t(&input, true);
            let loss = output.cross_entropy_for_logits(&target);

            opt.backward_step(&loss);

            total_loss += loss.double_value(&[]);
        }

        println!("DEBUG → Epoch {epoch} processed {} batches", batches);

        let avg_loss = total_loss / batches as f64;
        println!("Epoch {epoch}, avg train loss: {:.4}", avg_loss);

        // Sincronizza GPU prima di fermare tracker
        if device.is_cuda() {
            Cuda::synchronize(0);
        }
        stop_tracker();

        // --- Evaluation
        let eval_file = format!("vgg_cifar100_eval_epoch{:02}.csv", epoch);
        start_tracker("emissions", &eval_file);

        let mut correct = 0;
        let mut pred_class_hist = HashMap::new();
        let mut eval_batches = 0;

        for batch in test_data.chunks(batch_size) {
            eval_batches += 1;
            let images: Vec<_> = batch.iter().map(|(img, _)| img.unsqueeze(0)).collect();
            let labels: Vec<_> = batch.iter().map(|(_, label)| *label).collect();

            let input = Tensor::cat(&images, 0).to_device(device);
            let target = Tensor::from_slice(&labels)
                .to_kind(Kind::Int64)
                .to_device(device);

            let output = net.forward_t(&input, false);
            let predicted = output.argmax(-1, false);

            correct += predicted
                .eq_tensor(&target)
                .to_kind(Kind::Int64)
                .sum(Kind::Int64)
                .int64_value(&[]);

            for val in predicted.iter::<i64>().unwrap() {
                *pred_class_hist.entry(val).or_insert(0) += 1;
            }
        }

        println!("DEBUG → Epoch {epoch} processed {} eval batches", eval_batches);

        let accuracy = correct as f64 / test_data.len() as f64 * 100.0;
        println!("Epoch {epoch}, test accuracy: {:.2}%", accuracy);

        if pred_class_hist.len() <= 3 {
            println!(
                "⚠️ WARNING: Predicted classes are very few → possible class collapse: {:?}",
                pred_class_hist
            );
        }

        // Sincronizza GPU prima di fermare tracker
        if device.is_cuda() {
            Cuda::synchronize(0);
        }
        stop_tracker();
    }

    shutdown_tracker_daemon();
}
