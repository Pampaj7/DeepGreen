

use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use tch::nn::Module;

fn main() {
    // --- Setup ---
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let vs = nn::VarStore::new(device);
    let net = nn::seq()
        .add(nn::linear(&vs.root() / "layer1", 100, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "layer2", 64, 10, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    // --- Dummy data ---
    let input = Tensor::randn(&[64, 100], (tch::Kind::Float, device));
    let targets = Tensor::randint(10, &[64], (tch::Kind::Int64, device));
    // --- Training loop ---
    for epoch in 1..=5 {
        let logits = net.forward(&input);
        let loss = logits.cross_entropy_for_logits(&targets);
        opt.backward_step(&loss);

        println!("Epoch: {}, Loss: {:.4}", epoch, loss.double_value(&[]));
    }
}

