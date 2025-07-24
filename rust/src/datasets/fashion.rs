use std::collections::HashSet;
use std::fs;
use tch::{Tensor, vision::image, Device, Kind, Result};

pub fn load_fashion_mnist(dir: &str, device: Device) -> Result<Vec<(Tensor, i64)>> {
    let mut data = vec![];

    // Normalization constants for Fashion-MNIST (mean and std over 1 channel)
    let mean = Tensor::from_slice(&[0.2860])
        .to_kind(Kind::Float)
        .view([1, 1, 1])
        .to_device(device);
    let std = Tensor::from_slice(&[0.3530])
        .to_kind(Kind::Float)
        .view([1, 1, 1])
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
            if img_path.extension().and_then(|s| s.to_str()) != Some("png") {
                continue; // Salta file non PNG
            }

            let mut img = image::load(&img_path)?
                .to_device(device)
                .to_kind(Kind::Float) / 255.0;

            if img.size() == [28, 28] {
                img = img.unsqueeze(0); // Da [H, W] a [1, H, W]
            }

            let img = (img - &mean) / &std;
            data.push((img, class_id as i64));
        }

    }

    let unique_class_ids: HashSet<i64> = data.iter().map(|(_, cid)| *cid).collect();


    Ok(data)
}
