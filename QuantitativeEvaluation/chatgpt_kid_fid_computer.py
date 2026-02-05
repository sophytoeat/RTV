import numpy as np
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import cv2

# ---------- Step 1: Convert BGR to RGB ----------
def bgr_to_rgb(images):
    return [img[..., ::-1] for img in images]

# ---------- Step 2: Normalize and Resize ----------
def preprocess_images(images, size=(299, 299)):
    processed = []
    for img in images:
        img_rgb = cv2.resize(img, size)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1)  # HWC to CHW
        processed.append(img_tensor)
    return processed

# ---------- Step 3: Compute FID and KID ----------

def compute_fid_kid(real_images_bgr, fake_images_bgr, batch_size=32):
    import torch
    import cv2
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision import transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)

    # Transforms without normalization
    resize = transforms.Resize((299, 299))
    to_tensor_uint8 = transforms.Compose([
        transforms.ToPILImage(),
        resize,
        transforms.ToTensor(),  # This gives float, but we scale it back
        lambda x: (x * 255).to(torch.uint8)  # Convert to uint8
    ])

    def process_in_batches(images, is_real):
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensor_list = []

            for img in batch:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = to_tensor_uint8(img_rgb)
                tensor_list.append(tensor)

            tensor_batch = torch.stack(tensor_list).to(device)
            fid.update(tensor_batch, real=is_real)
            kid.update(tensor_batch, real=is_real)

    process_in_batches(real_images_bgr, is_real=True)
    process_in_batches(fake_images_bgr, is_real=False)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_score = kid_mean.item()
    return fid_score, kid_score
# ---------- Example Usage ----------
if __name__ == "__main__":
    # Example dummy data (replace with real images)
    real_images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(100)]
    fake_images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(100)]

    fid_score, kid_score = compute_fid_kid(real_images, fake_images)
    print(f"FID Score: {fid_score:.4f}")
    print(f"KID Score: {kid_score:.4f}")
