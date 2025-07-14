use tch::{nn, nn::OptimizerConfig, vision::resnet, Device, Kind, Tensor};
use tch::nn::ModuleT; // <--- IMPORTANTE

fn main() {
    // --- Device ---
    let device = if tch::Cuda::is_available() {
        println!("Using device: Cuda(0)");
        Device::Cuda(0)
    } else {
        println!("Using device: CPU");
        Device::Cpu
    };

    // --- Model ---
    let vs = nn::VarStore::new(device);
    let net = resnet::resnet18(&vs.root(), 100); // CIFAR100

    // --- Optimizer ---
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    // --- Training loop (dati finti) ---
    for epoch in 1..=5 {
        let input = Tensor::randn(&[64, 3, 32, 32], (Kind::Float, device));
        let targets = Tensor::randint(100, [64], (tch::Kind::Int64, device));

        let logits = net.forward_t(&input, true); // FIXATO
        let loss = logits.cross_entropy_for_logits(&targets);

        opt.backward_step(&loss);

        println!("Epoch: {}, Loss: {:.4}", epoch, loss.double_value(&[]));    }
}