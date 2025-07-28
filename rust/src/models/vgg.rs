use tch::{nn, nn::ModuleT};
use tch::{no_grad, Tensor};


pub fn vgg16<'a>(vs: &'a nn::Path<'a>, num_classes: i64) -> nn::FuncT<'a> {
    let features = nn::seq_t()
        .add(conv_block(&(vs / "conv1"), 3, 64, 2))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(conv_block(&(vs / "conv2"), 64, 128, 2))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(conv_block(&(vs / "conv3"), 128, 256, 3))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(conv_block(&(vs / "conv4"), 256, 512, 3))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(conv_block(&(vs / "conv5"), 512, 512, 3))
        .add_fn(|x| x.max_pool2d_default(2)); // [B, 512, 7, 7]

    let classifier = nn::seq_t()
        .add(nn::linear(vs / "fc1", 512 * 7 * 7, 4096, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| if train { x.dropout(0.5, train) } else { x.shallow_clone() })
        .add(nn::linear(vs / "fc2", 4096, 4096, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| if train { x.dropout(0.5, train) } else { x.shallow_clone() })
        .add(nn::linear(vs / "fc3", 4096, num_classes, Default::default()));

    nn::func_t(move |xs, train| {
        let out = xs
            .apply_t(&features, train)
            .view([-1, 512 * 3 * 3]);
        classifier.forward_t(&out, train)
    })
}

pub fn vgg16_tiny<'a>(vs: &'a nn::Path<'a>, num_classes: i64) -> nn::FuncT<'a> {
    let features = nn::seq_t()
        .add(conv_block(&(vs / "conv1"), 3, 64, 2))
        .add_fn(|x| x.max_pool2d_default(2))   // -> 32×32
        .add(conv_block(&(vs / "conv2"), 64, 128, 2))
        .add_fn(|x| x.max_pool2d_default(2))   // -> 16×16
        .add(conv_block(&(vs / "conv3"), 128, 256, 3))
        .add_fn(|x| x.max_pool2d_default(2))   // -> 8×8
        .add(conv_block(&(vs / "conv4"), 256, 512, 3))
        .add_fn(|x| x.max_pool2d_default(2))   // -> 4×4
        .add(conv_block(&(vs / "conv5"), 512, 512, 3))
        .add_fn(|x| x.max_pool2d_default(2));  // -> 2×2

    let classifier = nn::seq_t()
        .add(nn::linear(vs / "fc1", 512 * 2 * 2, 4096, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| if train { x.dropout(0.5, train) } else { x.shallow_clone() })
        .add(nn::linear(vs / "fc2", 4096, 4096, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| if train { x.dropout(0.5, train) } else { x.shallow_clone() })
        .add(nn::linear(vs / "fc3", 4096, num_classes, Default::default()));

    nn::func_t(move |xs, train| {
        let out = xs
            .apply_t(&features, train)
            .flatten(1, -1);  // flatten dinamico → 2048 per 64×64
        classifier.forward_t(&out, train)
    })
}

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


pub fn init_weights(vs: &nn::VarStore) {
    no_grad(|| {
        for (name, mut tensor) in vs.variables() {
            if name.ends_with("weight") {
                let std = (2.0 / tensor.size()[1] as f64).sqrt();
                let new_tensor = Tensor::empty_like(&tensor).uniform_(-std, std);
                let _ = tensor.copy_(&new_tensor);
            } else if name.ends_with("bias") {
                let new_tensor = Tensor::zeros_like(&tensor);
                let _ = tensor.copy_(&new_tensor);
            }
        }
    });
}