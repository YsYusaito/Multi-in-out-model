# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 11:31:04 2022

@author: 10087826
"""
import numpy as np
from PIL import Image
import torch
import tqdm


class DataSet:
    def __init__(self, data_path_list):
        
        self.img_data, self.hist_R, self.hist_G, self.hist_B,\
            self.label_1, self.label_2= self.data_loader(data_path_list)
            
        self.img_data = np.array(self.img_data)
        self.hist_R = np.array(self.hist_R)
        self.hist_G = np.array(self.hist_G)
        self.hist_B = np.array(self.hist_B)
        self.label_1 = np.array(self.label_1)
        self.label_2 = np.array(self.label_2)

        
    # train, test dataを読み込み、リストに格納(前処理も施しておく)
    def data_loader(self, data_path_list):
        
        img_data = []
        hist_R = []
        hist_G = []
        hist_B = []
        label_1 = []  # smiling or non smiling
        label_2 = []  # male or female
        
        mean_rgb = np.array([0.485, 0.456, 0.406])
        std_rgb = np.array([0.229, 0.224, 0.225])
        
        for path_img in tqdm.tqdm(data_path_list):
            
            img = self.load_image(path_img)
            
            img = np.array(img)
            img = img.transpose(2, 0, 1)  # ch, height, width
            
            img_float = np.zeros((img.shape[0],img.shape[1],img.shape[2]), dtype=np.float32)
            
            num_pixel = img.shape[1]*img.shape[2]
            
            for ch in range(3):
                
                if ch == 0:
                    hist_R.append(np.array((np.histogram(img[ch], bins=256)[0]/num_pixel), dtype=np.float32))
                elif ch==1:
                    hist_G.append(np.array((np.histogram(img[ch], bins=256)[0]/num_pixel), dtype=np.float32))
                else:
                    hist_B.append(np.array((np.histogram(img[ch], bins=256)[0]/num_pixel), dtype=np.float32))
                    
                img_float[ch] = (img[ch]/255 - mean_rgb[ch])/std_rgb[ch]
                
            img_data.append(img_float)
            
            label_num = str(path_img).split('\\')[1][:2]
            
            # male smiling
            if label_num == "00":
                label_1.append(0)
                label_2.append(0)
                
            # male nonsmiling
            elif label_num == "01":
                label_1.append(0)
                label_2.append(1)
                
            # female smiling
            elif label_num == "10":
                label_1.append(1)
                label_2.append(0)
                
            # female nonsmiling 11
            else:
                label_1.append(1)
                label_2.append(1)
                
        
        return img_data, hist_R, hist_G, hist_B, label_1, label_2
    
    def load_image(self, path_img, size=(224, 224)):
        img = Image.open(path_img)

        # 短辺長を基準とした正方形の座標を得る
        x_center = img.size[0] // 2
        y_center = img.size[1] // 2
        half_short_side = min(x_center, y_center)
        x0 = x_center - half_short_side
        y0 = y_center - half_short_side
        x1 = x_center + half_short_side
        y1 = y_center + half_short_side

        img = img.crop((x0, y0, x1, y1))
        img = img.resize(size)
        img = np.array(img, dtype=np.float32)
        return img 
    
    # ミニバッチをtensorにして返す
    def get_batch_tensor(self, get_range_list):
        
        return torch.from_numpy(self.img_data[get_range_list].astype(np.float32)).clone(), torch.from_numpy(self.hist_R[get_range_list].astype(np.float32)).clone(),\
            torch.from_numpy(self.hist_G[get_range_list].astype(np.float32)).clone(), torch.from_numpy(self.hist_B[get_range_list].astype(np.float32)).clone(),\
            torch.from_numpy(self.label_1[get_range_list].astype(np.int64)).clone(), torch.from_numpy(self.label_2[get_range_list].astype(np.int64)).clone()
    