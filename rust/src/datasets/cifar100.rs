use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use tch::{Tensor, vision::image, Device, Kind, Result};
use tch::vision::image::resize;

fn cifar100_norm(device: Device) -> (Tensor, Tensor) {
    let mean = Tensor::from_slice(&[0.5071, 0.4867, 0.4408])
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);
    let std = Tensor::from_slice(&[0.2675, 0.2565, 0.2761])
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);
    (mean, std)
}

pub struct Cifar100 {
    files: Vec<(PathBuf, i64)>,
    device: Device,
    resize_to: Option<i64>,
    mean: Tensor,
    std: Tensor,
}

impl Cifar100 {
    pub fn new(dir: &str, device: Device, resize_to: Option<i64>) -> Result<Self> {
        let (mean, std) = cifar100_norm(device);

        let mut class_folders: Vec<_> = fs::read_dir(dir)?.map(|e| e.unwrap().path()).collect();
        class_folders.sort_by_key(|p| p.file_name().unwrap().to_os_string());

        let mut files = vec![];
        for (class_id, class_path) in class_folders.into_iter().enumerate() {
            let mut images: Vec<_> = fs::read_dir(&class_path)?.map(|e| e.unwrap().path()).collect();
            images.sort();
            for img in images {
                files.push((img, class_id as i64));
            }
        }

        Ok(Self { files, device, resize_to, mean, std })
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

        if img.size() == [32, 32, 3] {
            img = img.permute(&[2, 0, 1]); // HWC -> CHW
        }

        if let Some(size) = self.resize_to {
            // resize va fatto su CPU, poi riportato al device target
            img = resize(&img.to(Device::Cpu), size, size)?.to(self.device);
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
                // usa indice diretto anziché position()
                let idx = self.files.iter().position(|x| &x.0 == path).unwrap();
                let (img, _) = self.load_item(idx)?;
                images.push(img.unsqueeze(0));
                labels.push(*label);
            }

            let x = Tensor::cat(&images, 0);
            let y = Tensor::from_slice(&labels).to_device(self.device);

            // Debug opzionale → commenta se non serve
            // println!("Batch loaded: x.shape={:?}, x.device={:?}", x.size(), x.device());

            Ok((x, y))
        })
    }
}
