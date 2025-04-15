import os
import random
import shutil

def downsample_dataset(source_dir, dest_dir, keep_per_class):
    classes = ["Fake", "Real"]
    os.makedirs(dest_dir, exist_ok=True)

    for cls in classes:
        src_folder = os.path.join(source_dir, cls)
        dst_folder = os.path.join(dest_dir, cls)
        os.makedirs(dst_folder, exist_ok=True)

        all_images = os.listdir(src_folder)
        if len(all_images) < keep_per_class:
            raise ValueError(f"Not enough images in {cls} to sample {keep_per_class}. Found {len(all_images)}.")

        sampled_images = random.sample(all_images, keep_per_class)
        for img in sampled_images:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

        print(f"✅ Copied {keep_per_class} images from '{cls}' to '{dst_folder}'")
