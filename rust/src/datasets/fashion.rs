use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use tch::{Tensor, vision::image, Device, Kind, Result};

fn fashion_norm(device: Device) -> (Tensor, Tensor) {
    let mean = Tensor::from_slice(&[0.2860])
        .to_kind(Kind::Float)
        .view([1, 1, 1])
        .to_device(device);
    let std = Tensor::from_slice(&[0.3530])
        .to_kind(Kind::Float)
        .view([1, 1, 1])
        .to_device(device);
    (mean, std)
}

pub struct Fashion {
    files: Vec<(PathBuf, i64)>,
    device: Device,
    mean: Tensor,
    std: Tensor,
}

impl Fashion {
    pub fn new(dir: &str, device: Device, _resize_to: Option<i64>) -> Result<Self> {
        let (mean, std) = fashion_norm(device);

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

        Ok(Self { files, device, mean, std })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn shuffle<R: rand::Rng>(&mut self, rng: &mut R) {
        self.files.shuffle(rng);
    }

    fn load_item(&self, idx: usize) -> Result<(Tensor, i64)> {
        let (ref path, label) = self.files[idx];
        let mut img = image::load(path)?.to_kind(Kind::Float) / 255.0;

        if img.size() == [28, 28] {
            img = img.unsqueeze(0); // [1, H, W]
        }

        let img = (img.to_device(self.device) - &self.mean) / &self.std;
        Ok((img, label))
    }

    pub fn iter_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        self.files.chunks(batch_size).map(move |chunk| {
            let mut images = Vec::with_capacity(chunk.len());
            let mut labels = Vec::with_capacity(chunk.len());

            for (path, label) in chunk {
                let idx = self.files.iter().position(|x| &x.0 == path).unwrap();
                let (img, _) = self.load_item(idx)?;
                images.push(img.unsqueeze(0));
                labels.push(*label);
            }

            let x = Tensor::cat(&images, 0);
            let y = Tensor::from_slice(&labels).to_device(self.device);
            Ok((x, y))
        })
    }
}
