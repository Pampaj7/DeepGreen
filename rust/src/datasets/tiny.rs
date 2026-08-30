use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use tch::{Tensor, vision::image, Device, Kind, Result};
use tch::vision::image::resize;
use rayon::iter::IntoParallelRefIterator;  // questo mancava

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
        crate::init_loader_pool();
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
        // image::load returns a *uint8* tensor already in [C,H,W] order, not
        // [H,W,C]. The previous narrow(2, 0, 3) therefore cropped the *width*
        // to three pixels and the permute that followed transposed the result,
        // so this loader fed the network a 3-pixel-wide sliver of each image.
        // The mistake never surfaced as a wrong accuracy because resize()
        // returns uint8 and the run died on the dtype first.
        let mut img = image::load(path)?;

        if img.size()[0] > 3 {
            img = img.narrow(0, 0, 3); // drop the alpha channel
        } else if img.size()[0] == 1 {
            img = img.repeat(&[3, 1, 1]); // Tiny ImageNet holds some greyscale images
        }

        // Resize opzionale -- on the raw uint8: resize() takes and returns
        // uint8, so converting to float before it discards the scaling.
        // Only when it is not already that size. resize() resamples in uint8,
        // so calling it at the target resolution is not the identity: it moved
        // the pixel standard deviation 1.3% away from every other stack on a
        // dataset that, since the images are pre-resized on disk, needs no
        // resizing at all.
        if let Some(size) = self.resize_to {
            if img.size()[1] != size || img.size()[2] != size {
                img = resize(&img, size, size)?;
            }
        }

        let img = img.to_kind(Kind::Float) / 255.0;

        // Normalizzazione e device
        // Normalisation is off by default: the other seven ecosystems feed
        // raw [0,1] inputs. See crate::normalize_inputs().
        let img = if crate::normalize_inputs() {
            (img.to_device(self.device) - &self.mean) / &self.std
        } else {
            img.to_device(self.device)
        };
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
