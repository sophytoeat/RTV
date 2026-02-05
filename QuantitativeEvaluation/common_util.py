import os
import numpy
import numpy as np
import torch
from util.video_loader import VideoLoader

def extract_video_feature_from_np_list(model, image_list, cuda=True):
    frame_tensors = []
    for frame in image_list:
        frame_np = frame
        frame_tensors.append(np2tensor(frame_np))
    if cuda:
        model_input = torch.stack(frame_tensors, dim=2).cuda()
    else:
        model_input = torch.stack(frame_tensors, dim=2)
    with torch.no_grad():
        pred = model.extract_features(model_input, target_endpoints='Logits')
    # Obtain global spatially-pooled features as single vector
    features = pred[0].squeeze()  # 1024
    return features.cpu().numpy()

def extract_video_feature(model, video_path):
    video_loader = VideoLoader(video_path)
    frame_tensors = []
    for i in range(len(video_loader)):
        frame_np = video_loader.get_raw_numpy_image(i)
        frame_tensors.append(np2tensor(frame_np))
    model_input = torch.stack(frame_tensors, dim=2).cuda()
    with torch.no_grad():
        pred = model.extract_features(model_input, target_endpoints='Logits')
    # Obtain global spatially-pooled features as single vector
    features = pred[0].squeeze()  # 1024
    return features.cpu().numpy()



def np2tensor(img: np.ndarray) -> torch.tensor:
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
