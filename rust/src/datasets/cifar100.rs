use std::collections::HashSet;
use std::fs;
use tch::{Tensor, vision::image, Device, Kind, Result};
use tch::vision::image::resize;

pub fn load_cifar100(dir: &str, device: Device, resize_to: Option<i64>) -> Result<Vec<(Tensor, i64)>> {
    let mut data = vec![];

    // Normalization constants (CIFAR-100)
    let mean = Tensor::f_from_slice(&[0.5071, 0.4867, 0.4408])?
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);
    let std = Tensor::f_from_slice(&[0.2675, 0.2565, 0.2761])?
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);

    let mut class_folders: Vec<_> = fs::read_dir(dir)?
        .map(|e| e.unwrap().path())
        .collect();
    class_folders.sort_by_key(|path| path.file_name().unwrap().to_os_string());

    for (class_id, class_path) in class_folders.into_iter().enumerate() {
        println!("Reading class_id: {} from {:?}", class_id, class_path.file_name());

        let mut images: Vec<_> = fs::read_dir(&class_path)?
            .map(|e| e.unwrap().path())
            .collect();
        images.sort();

        if images.is_empty() {
            println!("⚠️ Warning: no images found in {:?}", class_path);
        }

            for img_path in images {
                let mut img = image::load(&img_path)?
                    .to_kind(Kind::Float) / 255.0;

                if img.size() == [32, 32, 3] {
                    img = img.permute(&[2, 0, 1]);
                }

                if let Some(size) = resize_to {
                    img = resize(&img.to(Device::Cpu), size, size)?.to(device); // resize va fatto su CPU
                }


                let img = img.to_device(device);
                let img = (img - &mean) / &std;

                data.push((img, class_id as i64));
            }

    }

    let unique_class_ids: HashSet<i64> = data.iter().map(|(_, cid)| *cid).collect();
    assert_eq!(
        unique_class_ids.len(),
        100,
        "Expected 100 unique class IDs, found {}",
        unique_class_ids.len()
    );

    Ok(data)
}
