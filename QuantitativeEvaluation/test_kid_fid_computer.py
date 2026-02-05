import os
import shutil
import tempfile
import numpy as np
import cv2
from cleanfid import fid

def save_images_to_folder(images, folder):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(folder, f"{i:05d}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def compute_clean_fid_kid(real_images_rgb, fake_images_rgb):
    # Create temporary directories
    real_dir = tempfile.mkdtemp(prefix="real_")
    fake_dir = tempfile.mkdtemp(prefix="fake_")

    try:
        # Save images to folders
        save_images_to_folder(real_images_rgb, real_dir)
        save_images_to_folder(fake_images_rgb, fake_dir)

        # Compute FID and KID
        fid_score = fid.compute_fid(real_dir, fake_dir)
        kid_score = fid.compute_kid(real_dir, fake_dir)

        return fid_score, kid_score

    finally:
        # Clean up temporary directories
        shutil.rmtree(real_dir)
        shutil.rmtree(fake_dir)

# Example usage
if __name__ == "__main__":
    # Generate dummy identical RGB images for testing
    real_images = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(100)]
    fake_images = real_images.copy()

    fid_score, kid_score = compute_clean_fid_kid(real_images, fake_images)
    print(f"Clean FID Score: {fid_score:.4f}")
    print(f"Clean KID Score: {kid_score:.6f}")
