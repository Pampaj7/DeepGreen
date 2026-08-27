use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use rayon::prelude::*; // parallelismo CPU
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
        crate::init_loader_pool();
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

    pub fn iter_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        self.files.chunks(batch_size).map(move |chunk| {
            // Carichiamo le immagini in parallelo su CPU → qui NON creiamo Tensor
            let samples: Result<Vec<(Vec<u8>, i64)>> = chunk
                .par_iter()
                .map(|(path, label)| {
                    let img_buf = std::fs::read(path)?; // raw bytes
                    Ok((img_buf, *label))
                })
                .collect();

            let samples = samples?;

            let mut images = Vec::with_capacity(samples.len());
            let mut labels = Vec::with_capacity(samples.len());

            for (img_buf, label) in samples {
                // Decodifica PNG → Tensor CPU
                // Same operation order as fashion.rs and tiny.rs: decode to
                // uint8 [C,H,W], fix the shape, resize, and only then scale to
                // [0,1]. resize() takes and returns uint8, so a float tensor
                // handed to it comes back as bytes. CIFAR-100 passes
                // resize_to = None today and so never hit that, but the order
                // is what kept it working, not the dataset.
                let mut img = image::load_from_memory(&img_buf)?;

                if img.size()[0] > 3 {
                    img = img.narrow(0, 0, 3); // drop the alpha channel
                }

                if let Some(size) = self.resize_to {
                    img = resize(&img.to(Device::Cpu), size, size)?;
                }

                let img = img.to_kind(Kind::Float) / 255.0;

                images.push(img.unsqueeze(0));
                labels.push(label);
            }

            // Cat e normalizzazione su GPU
            let mut x = Tensor::cat(&images, 0).to_device(self.device);
            // Normalisation is off by default: the other seven ecosystems feed
            // raw [0,1] inputs. See crate::normalize_inputs().
            if crate::normalize_inputs() {
                x = (x - &self.mean) / &self.std;
            }

            let y = Tensor::from_slice(&labels).to_device(self.device);

            Ok((x, y))
        })
    }
}
