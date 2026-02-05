import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from util.gdrive_downloader import download_gdrive_file
from tqdm import tqdm
import cv2

from QuantitativeEvaluation.fid_metrics import (
    ImageDataset,
    ImageSequenceDataset,
    VideoDataset,
    build_inception,
    build_inception3d,
    build_resnet3d,
    calculate_fid,
    is_image_dir_path,
    is_video_path,
    postprocess_i2d_pred,
)


#todo: set clip size to 10, currently as 1
class VfidComputerI3D:
    def __init__(self):
        self.path = './pretrained_models/i3d.pt'
        self.gid = '1AUTu5cbou7JWxeBP_BL9V12M1jhNWQDd'
        self.clip_length = 10
        download_gdrive_file(self.path, self.gid)
        self.model = build_inception3d(self.path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def __call__(self, image_list1, image_list2):
        feat0 = []
        i = 0
        sub_list = []
        for img in tqdm(image_list1):
            sub_list.append(img)
            if i % self.clip_length == self.clip_length - 1:
                feat0.append(self.compute_feature(sub_list).cpu().numpy())
                sub_list=[]
            i = i + 1

        feat0 = np.concatenate(feat0, axis=0)
        feat1 = []
        i=0
        sub_list = []
        for img in tqdm(image_list2):
            sub_list.append(img)
            if i % self.clip_length == self.clip_length - 1:
                feat1.append(self.compute_feature(sub_list).cpu().numpy())
                sub_list = []
            i = i + 1
        feat1 = np.concatenate(feat1, axis=0)
        return calculate_fid(feat0, feat1)

    def compute_feature(self, sub_list: list):
        tensor_list=[]
        for img in sub_list:
            img = cv2.resize(img, (224, 224))
            img = img[:, :, [2, 1, 0]]
            img = img.astype(np.float32) / 255.0
            img = img * 2.0 - 1.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)#BCHW
            if torch.cuda.is_available():
                img = img.cuda()
            img = img.unsqueeze(2)  # add time dimension
            tensor_list.append(img)
        input_tensor=torch.cat(tensor_list, dim=2)
        #img=torch.cat((img,)*16,dim=2)
        #print(img.shape)
        with torch.no_grad():
            pred = self.model.extract_features(input_tensor)
            #print(pred.shape)
        pred = pred.mean(4).mean(3).mean(2)
        #pred = pred.squeeze(3).squeeze(3).mean(2)
        #print(pred.shape)
        return pred


class VfidComputerResnext:
    def __init__(self):
        self.path = './pretrained_models/resnext-101.pth'
        self.gid = '1I5OJ3WU4UfI78UZCpC5hK5HsYpHohQvx'
        download_gdrive_file(self.path, self.gid)
        self.model = build_resnet3d(self.path)
        self.clip_length = 10
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def __call__old(self, image_list1, image_list2):
        feat0 = []
        for img in image_list1:
            feat0.append(self.compute_feature(img).cpu().numpy())
        feat0 = np.concatenate(feat0, axis=0)
        feat1 = []
        for img in image_list2:
            feat1.append(self.compute_feature(img).cpu().numpy())
        feat1 = np.concatenate(feat1, axis=0)
        return calculate_fid(feat0, feat1)

    def __call__(self, image_list1, image_list2):
        feat0 = []
        i = 0
        sub_list = []
        for img in tqdm(image_list1):
            sub_list.append(img)
            if i % self.clip_length == self.clip_length - 1:
                feat0.append(self.compute_feature(sub_list).cpu().numpy())
                sub_list = []
            i = i + 1

        feat0 = np.concatenate(feat0, axis=0)
        feat1 = []
        i = 0
        sub_list = []
        for img in tqdm(image_list2):
            sub_list.append(img)
            if i % self.clip_length == self.clip_length - 1:
                feat1.append(self.compute_feature(sub_list).cpu().numpy())
                sub_list = []
            i = i + 1
        feat1 = np.concatenate(feat1, axis=0)
        return calculate_fid(feat0, feat1)

    def compute_feature_old(self, img: np.ndarray):
        img = cv2.resize(img, (224, 224))
        img = img[:, :, [2, 1, 0]]
        img = img.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        img = img.unsqueeze(2)
        with torch.no_grad():
            pred = self.model.extract_features(img)
            #print(pred.shape)
        return pred

    def compute_feature(self, sub_list: list):
        tensor_list=[]
        for img in sub_list:
            img = cv2.resize(img, (224, 224))
            img = img[:, :, [2, 1, 0]]
            img = img.astype(np.float32) / 255.0
            img = img * 2.0 - 1.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)#BCHW
            if torch.cuda.is_available():
                img = img.cuda()
            img = img.unsqueeze(2)  # add time dimension
            tensor_list.append(img)
        input_tensor=torch.cat(tensor_list, dim=2)
        #img=torch.cat((img,)*16,dim=2)
        #print(img.shape)
        with torch.no_grad():
            pred = self.model.extract_features(input_tensor)
            #print(pred.shape)
        #pred = pred.mean(4).mean(3).mean(2)
        #pred = pred.squeeze(3).squeeze(3).mean(2)
        #print(pred.shape)
        return pred
