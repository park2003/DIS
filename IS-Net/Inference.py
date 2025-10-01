import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *


if __name__ == "__main__":
    dataset_path="../demo_datasets/your_dataset/sofa2.jpg"  #Your dataset path
    model_path="../saved_models/IS-Net/isnet-general-use.pth"  # the model path
    result_path="../demo_datasets/your_dataset_result/test"  #The folder path that you want to save the results
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    im_list = [dataset_path] if os.path.isfile(dataset_path) else glob(os.path.join(dataset_path, '*'))
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear", align_corners=False).type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            
            result = net(image)
            result = torch.squeeze(F.interpolate(result[0][0],im_shp,mode='bilinear', align_corners=False))
            
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)

            alpha_mask = result.cpu().numpy()[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            if im.shape[2] > 3:
                im = im[:,:,:3]

            white_bg = np.full_like(im, 255, dtype=np.uint8)
            final_image = (im * alpha_mask + white_bg * (1 - alpha_mask)).astype(np.uint8)

            im_name, _ = os.path.splitext(os.path.basename(im_path))
            io.imsave(os.path.join(result_path,im_name+".png"), final_image)
