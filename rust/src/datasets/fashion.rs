use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use rayon::prelude::*;
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

        let mut class_folders: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|x| x.path()))
            .collect();
        class_folders.sort_by_key(|p| p.file_name().unwrap().to_os_string());

        let mut files = vec![];
        for (class_id, class_path) in class_folders.into_iter().enumerate() {
            let mut images: Vec<_> = fs::read_dir(&class_path)?
                .filter_map(|e| e.ok().map(|x| x.path()))
                .collect();
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

    pub fn iter_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        self.files.chunks(batch_size).map(move |chunk| {
            // carichiamo in parallelo solo su CPU
            let samples: Result<Vec<(Tensor, i64)>> = chunk
                .par_iter()
                .map(|(path, label)| {
                    let mut img = image::load(path)?.to_kind(Kind::Float) / 255.0;

                    if img.size().len() == 2 {
                        img = img.unsqueeze(0); // [1,H,W]
                    } else if img.size()[2] == 1 {
                        img = img.permute(&[2, 0, 1]); // [1,H,W]
                    }

                    // NON .to_device qui, restiamo su CPU
                    Ok((img, *label))
                })
                .collect();

            let samples = samples?;

            let images: Vec<Tensor> = samples.iter().map(|(x, _)| x.unsqueeze(0)).collect();
            let labels: Vec<i64> = samples.iter().map(|(_, y)| *y).collect();

            // concateno e poi sposto su device + normalizzo
            let mut x = Tensor::cat(&images, 0).to_device(self.device);
            x = (x - &self.mean) / &self.std;

            let y = Tensor::from_slice(&labels).to_device(self.device);

            Ok((x, y))
        })
    }

}
