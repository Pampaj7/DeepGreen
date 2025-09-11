use tch::{nn, nn::ModuleT};

/// VGG16 conforme a torchvision, adattato per input 32×32 (es. CIFAR datasets).
/// Differisce solo nell’ultimo layer del classifier (4096 → num_classes).
pub fn vgg16<'a>(vs: &'a nn::Path<'a>, num_classes: i64) -> nn::FuncT<'a> {
    let features = nn::seq_t()
        // Block 1
        .add(conv_block(&(vs / "conv1"), 3, 64, 2))
        .add_fn(|x| x.max_pool2d_default(2))
        // Block 2
        .add(conv_block(&(vs / "conv2"), 64, 128, 2))
        .add_fn(|x| x.max_pool2d_default(2))
        // Block 3
        .add(conv_block(&(vs / "conv3"), 128, 256, 3))
        .add_fn(|x| x.max_pool2d_default(2))
        // Block 4
        .add(conv_block(&(vs / "conv4"), 256, 512, 3))
        .add_fn(|x| x.max_pool2d_default(2))
        // Block 5
        .add(conv_block(&(vs / "conv5"), 512, 512, 3))
        .add_fn(|x| x.max_pool2d_default(2)); // [B, 512, 1, 1] su input 32×32

    let classifier = nn::seq_t()
        .add(nn::linear(vs / "fc1", 512, 4096, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| if train { x.dropout(0.5, true) } else { x.shallow_clone() })
        .add(nn::linear(vs / "fc2", 4096, 4096, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| if train { x.dropout(0.5, true) } else { x.shallow_clone() })
        .add(nn::linear(vs / "fc3", 4096, num_classes, Default::default())); // ← modificato

    nn::func_t(move |xs, train| {
        let out = xs.apply_t(&features, train).flatten(1, -1);
        classifier.forward_t(&out, train)
    })
}

/// Blocchi conv3x3 + ReLU × num_convs (standard VGG).
fn conv_block(vs: &nn::Path, in_c: i64, out_c: i64, num_convs: usize) -> nn::SequentialT {
    let mut seq = nn::seq_t();
    let mut input = in_c;
    for i in 0..num_convs {
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        seq = seq
            .add(nn::conv2d(&(vs / format!("conv{}", i)), input, out_c, 3, conv_cfg))
            .add_fn(|x| x.relu());
        input = out_c;
    }
    seq
}
