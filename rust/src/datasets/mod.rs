pub mod cifar100;
pub mod fashion;
pub mod imagenette;
pub mod tiny;

pub use crate::datasets::cifar100::Cifar100;
pub use crate::datasets::fashion::Fashion;
pub use crate::datasets::imagenette::Imagenette;
pub use crate::datasets::tiny::TinyImageNet;
