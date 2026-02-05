from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import lpips
import os
import wget
from QuantitativeEvaluation.I3D.pytorch_i3d import InceptionI3d
from QuantitativeEvaluation.common_util import extract_video_feature, extract_video_feature_from_np_list
from .frechet_distance import calculate_frechet_distance
from QuantitativeEvaluation.ResNeXt.resnext import generate_model




class SsimComputer:
    def __init__(self):
        pass

    def forward(self, img1:np.ndarray, img2:np.ndarray):
        #print(img1.shape)
        #print(img2.shape)
        return ssim(img1,img2,channel_axis=-1)

    def __call__(self, img_list1,img_list2):
        n=len(img_list1)
        score_sum=0
        for i in range(n):
            score_sum+=self.forward(img_list1[i],img_list2[i])
        return score_sum/n


class LpipsComputer:
    def __init__(self):
        self.lpips_model = lpips.LPIPS(net='alex')  # or vgg

    def forward(self, img1:np.ndarray, img2:np.ndarray):
        img1_tensor = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float() / 255.0
        distance = self.lpips_model(img1_tensor,img2_tensor)
        return distance

    def __call__(self, img_list1,img_list2):
        n=len(img_list1)
        score_sum=0
        for i in range(n):
            score_sum+=self.forward(img_list1[i],img_list2[i]).item()
        return score_sum/n

class VfidComputer:
    def __init__(self,cuda=True):
        #rgb_imagenet.pt https://github.com/piergiaj/pytorch-i3d/raw/eb3580bc5a9f3f7dd07d3162ed1d9674581ed3a5/models/rgb_imagenet.pt
        self.model_url = 'https://github.com/piergiaj/pytorch-i3d/raw/eb3580bc5a9f3f7dd07d3162ed1d9674581ed3a5/models/rgb_imagenet.pt'
        self.pt_name = 'rgb_imagenet.pt'
        self.pt_path = 'pretrained_models'
        self.download_pretrained()
        self.model = InceptionI3d(400, in_channels=3)
        self.weights = torch.load(os.path.join(self.pt_path,self.pt_name))
        self.model.load_state_dict(self.weights)
        if cuda:
            self.model.cuda()
        self.model.eval()
        self.use_cuda=cuda

    def download_pretrained(self):
        if os.path.exists(os.path.join(self.pt_path, self.pt_name)):
            return
        wget.download(self.model_url, out=self.pt_path)

    def __call__(self, image_list1, image_list2):
        feature1 = extract_video_feature_from_np_list(self.model,image_list1,cuda=self.use_cuda)
        feature2 = extract_video_feature_from_np_list(self.model,image_list2,cuda=self.use_cuda)
        mu1 = np.mean(feature1)
        sigma1=np.cov(feature1)
        mu2 = np.mean(feature2)
        sigma2 = np.cov(feature2)
        return calculate_frechet_distance(mu1,sigma1,mu2,sigma2)

    def forward(self, video_path1, video_path2):
        feature1 = extract_video_feature(self.model,video_path1)
        feature2 = extract_video_feature(self.model,video_path2)
        mu1 = np.mean(feature1)
        sigma1=np.cov(feature1)
        mu2 = np.mean(feature2)
        sigma2 = np.cov(feature2)
        return calculate_frechet_distance(mu1,sigma1,mu2,sigma2)









