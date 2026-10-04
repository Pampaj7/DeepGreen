use std::fs;
use std::path::PathBuf;
use rand::seq::SliceRandom;
use rayon::prelude::*; // parallelismo CPU
use tch::{Tensor, vision::image, Device, Kind, Result};
use tch::vision::image::resize;

/// Normalizzazione standard di ImageNet.
///
/// Kept only so that `DEEPGREEN_NORMALIZE=1` means the same thing here as in
/// the other three loaders. It is off by default (see `crate::normalize_inputs`):
/// the saturation cell feeds raw [0,1] inputs like every other stack.
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

/// Imagenette: ten ImageNet classes, 224x224x3 RGB PNG on disk.
///
/// Only the file list lives in the struct. Nothing is decoded or moved to the
/// accelerator until `iter_batches` reaches it, which is what makes 224x224
/// affordable: the whole train split as one float tensor would be
/// 9,469 x 3 x 224 x 224 x 4 B = 5.3 GiB resident before the model is loaded,
/// against 18 MiB for one batch of 32. `Cifar100` and `TinyImageNet` hold
/// their data the same way; this loader keeps that and does not preload.
pub struct Imagenette {
    files: Vec<(PathBuf, i64)>,
    classes: Vec<String>,
    device: Device,
    resize_to: Option<i64>,
    mean: Tensor,
    std: Tensor,
}

impl Imagenette {
    pub fn new(dir: &str, device: Device, resize_to: Option<i64>) -> Result<Self> {
        crate::init_loader_pool();
        let (mean, std) = imagenet_norm(device);

        // Class id = rank of the directory name in a plain byte-wise sort,
        // which is what torchvision's ImageFolder, the C++ ImageFolder and the
        // Java loader all do. `OsString`'s ordering on Unix *is* the byte
        // ordering of the raw name, so this is locale-independent -- an
        // `LC_COLLATE` that ignores case or spaces cannot move a class here.
        // The directory names hold spaces ("cassette player", "chain saw") and
        // are all lowercase since the converter stopped emitting the two
        // capitalised ImageNet names, so byte order and the obvious
        // alphabetical order now agree:
        //   0 cassette player   1 chain saw    2 church      3 english springer
        //   4 french horn       5 garbage truck  6 gas pump  7 golf ball
        //   8 parachute         9 tench
        // Getting this wrong permutes the labels of one stack against the
        // others and shows up only as a stack that never learns.
        let mut class_folders: Vec<_> = fs::read_dir(dir)?.map(|e| e.unwrap().path()).collect();
        class_folders.sort_by_key(|p| p.file_name().unwrap().to_os_string());
        let classes: Vec<String> = class_folders
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

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

        Ok(Self { files, classes, device, resize_to, mean, std })
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Class names in label order, so a run can record the mapping it used
    /// rather than leaving it to be inferred from a directory listing taken
    /// later, under another locale.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn shuffle<R: rand::Rng>(&mut self, rng: &mut R) {
        self.files.shuffle(rng);
    }

    pub fn iter_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        // Bound out of the struct so the decoding closure captures a plain
        // Option<i64> and not `&Self`, which holds Tensors.
        let resize_to = self.resize_to;
        self.files.chunks(batch_size).map(move |chunk| {
            // Read AND decode inside the rayon pool, which
            // `crate::init_loader_pool()` has bounded to DEEPGREEN_LOADER_THREADS
            // (2 by campaign contract, S3). Cifar100 parallelises the read and
            // decodes on the calling thread, which is invisible at 32x32 --
            // a 3 KiB PNG decodes in tens of microseconds -- but at 224x224 the
            // decode is ~50x larger and single-threaded decoding would make
            // this cell loader-bound, which is the one thing the saturation
            // contrast must not be. Two decoding threads is what PyTorch's
            // num_workers=2, the C++ .workers(2) and DL4J's
            // AsyncDataSetIterator(_, 2) each give their own stack.
            let samples: Result<Vec<(Tensor, i64)>> = chunk
                .par_iter()
                .map(|(path, label)| {
                    let img_buf = std::fs::read(path)?; // raw bytes

                    // Same operation order as fashion.rs, cifar100.rs and
                    // tiny.rs: decode to uint8 [C,H,W], fix the shape, resize,
                    // and only then scale to [0,1]. resize() takes and returns
                    // uint8, so a float tensor handed to it comes back as
                    // bytes.
                    let mut img = image::load_from_memory(&img_buf)?;

                    if img.size()[0] > 3 {
                        img = img.narrow(0, 0, 3); // drop the alpha channel
                    } else if img.size()[0] == 1 {
                        img = img.repeat(&[3, 1, 1]); // a greyscale JPEG survived the conversion
                    }

                    // Guarded, and a no-op for this dataset: the binaries pass
                    // resize_to = None because dataloader/download_convert_imagenette.py
                    // already wrote every image at 224x224 (spec S3: no stack
                    // resizes at run time). The guard is kept in the same form
                    // as cifar100.rs so that the four loaders read alike --
                    // resize() resamples in uint8 and rounds, so calling it at
                    // the target resolution is not the identity and moved this
                    // stack's pixel standard deviation off every other stack's
                    // the last time it ran unguarded.
                    if let Some(size) = resize_to {
                        if img.size()[1] != size || img.size()[2] != size {
                            img = resize(&img.to(Device::Cpu), size, size)?;
                        }
                    }

                    let img = img.to_kind(Kind::Float) / 255.0;
                    Ok((img.unsqueeze(0), *label))
                })
                .collect();

            let samples = samples?;

            let mut images = Vec::with_capacity(samples.len());
            let mut labels = Vec::with_capacity(samples.len());
            for (img, label) in samples {
                images.push(img);
                labels.push(label);
            }

            // One host-to-device copy per batch (18 MiB at batch 32), not one
            // per image: the concatenation happens on the CPU and the batch
            // crosses PCIe once.
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
