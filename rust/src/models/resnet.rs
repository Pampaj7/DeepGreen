use tch::{nn, vision::resnet};

pub fn resnet18<'a>(vs: &'a nn::Path<'a>, num_classes: i64) -> nn::FuncT<'a> {
    resnet::resnet18(vs, num_classes)
}
