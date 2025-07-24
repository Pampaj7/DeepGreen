use std::collections::HashSet;
use std::fs;
use tch::{Tensor, vision::image, Device, Kind, Result};

pub fn load_tiny_imagenet(dir: &str, device: Device) -> Result<Vec<(Tensor, i64)>> {
    let mut data = vec![];

    // Normalization constants (standard ImageNet)
    let mean = Tensor::f_from_slice(&[0.485, 0.456, 0.406])?
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);
    let std = Tensor::f_from_slice(&[0.229, 0.224, 0.225])?
        .to_kind(Kind::Float)
        .view([3, 1, 1])
        .to_device(device);

    // Leggi le cartelle delle classi e ordinale in modo deterministico
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
                .to_device(device)
                .to_kind(Kind::Float) / 255.0;

            // Trasponi da [H, W, C] a [C, H, W] se necessario
            if img.size() == [64, 64, 3] {
                img = img.permute(&[2, 0, 1]);
            } else if img.size() == [3, 64, 64] {
                // Già nel formato corretto
            } else {
                println!("⚠️ Unexpected image size: {:?} for {:?}", img.size(), img_path);
                continue;
            }

            let img = (img - &mean) / &std;
            data.push((img, class_id as i64));
        }
    }

    // Verifica che ci siano esattamente 200 classi
    let unique_class_ids: HashSet<i64> = data.iter().map(|(_, cid)| *cid).collect();
    assert_eq!(
        unique_class_ids.len(),
        200,
        "Expected 200 unique class IDs, found {}",
        unique_class_ids.len()
    );

    Ok(data)
}
