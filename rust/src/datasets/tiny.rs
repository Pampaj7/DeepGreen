use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use tch::{Tensor, vision::image, Device, Kind, Result};
use tch::vision::image::resize;

/// Normalizzazione standard di ImageNet
fn imagenet_norm(device: Device) -> (Tensor, Tensor) {
    let mean = Tensor::from_slice(&[0.485, 0.456, 0.406])
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);
    let std = Tensor::from_slice(&[0.229, 0.224, 0.225])
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);
    (mean, std)
}

/// Struct per TinyImageNet
pub struct TinyImageNet {
    files: Vec<(PathBuf, i64)>,
    device: Device,
    mean: Tensor,
    std: Tensor,
    resize_to: Option<i64>,
}

impl TinyImageNet {
    /// Costruttore: carica path immagini e setta normalizzazione
    pub fn new(dir: &str, device: Device, resize_to: Option<i64>) -> Result<Self> {
        let (mean, std) = imagenet_norm(device);

        let mut class_folders: Vec<_> = fs::read_dir(dir)?.map(|e| e.unwrap().path()).collect();
        class_folders.sort_by_key(|p| p.file_name().unwrap().to_os_string());

        let mut files = vec![];
        for (class_id, class_path) in class_folders.into_iter().enumerate() {
            let mut images: Vec<_> = fs::read_dir(&class_path)?.map(|e| e.unwrap().path()).collect();
            images.sort();
            for img in images {
                if img.extension().and_then(|s| s.to_str()) == Some("png") {
                    files.push((img, class_id as i64));
                }
            }
        }

        Ok(Self { files, device, mean, std, resize_to })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn shuffle<R: rand::Rng>(&mut self, rng: &mut R) {
        self.files.shuffle(rng);
    }

    /// Carica una singola immagine da path
    fn load_item_by_path(&self, path: &PathBuf, label: i64) -> Result<(Tensor, i64)> {
        // Carica immagine da file → float [H,W,C]
        let mut img = image::load(path)?.to_kind(Kind::Float) / 255.0;

        // HWC → CHW
        img = img.permute(&[2, 0, 1]);

        // Resize opzionale
        let img_cpu = if let Some(size) = self.resize_to {
            resize(&img, size, size)?
        } else {
            img
        };

        // Normalizzazione e device
        let img = (img_cpu.to_device(self.device) - &self.mean) / &self.std;
        Ok((img, label))
    }

    /// Itera in batch
    pub fn iter_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        self.files.chunks(batch_size).map(move |chunk| {
            let mut images = Vec::with_capacity(chunk.len());
            let mut labels = Vec::with_capacity(chunk.len());

            for (path, label) in chunk {
                let (img, _) = self.load_item_by_path(path, *label)?;
                images.push(img.unsqueeze(0));
                labels.push(*label);
            }

            let x = Tensor::cat(&images, 0);
            let y = Tensor::from_slice(&labels).to_device(self.device);
            Ok((x, y))
        })
    }
}
