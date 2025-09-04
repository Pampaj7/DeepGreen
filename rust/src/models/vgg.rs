// src/models/vgg32.rs
use tch::{nn, nn::ModuleT, Device, Kind, Tensor, vision};


pub struct Vgg16Stock32<'a> {
    inner: nn::FuncT<'a>,
}

impl<'a> Vgg16Stock32<'a> {

    pub fn new(vs: &'a nn::Path<'a>, num_classes: i64) -> Self {
        let inner = tch::vision::vgg::vgg16(vs, num_classes);
        Self { inner }
    }
}

impl<'a> nn::ModuleT for Vgg16Stock32<'a> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // xs atteso: [N, 3, 32, 32], float32 (preferibilmente normalizzato)
        self.inner.forward_t(xs, train)
    }
}