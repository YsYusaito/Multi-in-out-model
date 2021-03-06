import torch
import torchvision.models as models
import numpy as np
import tqdm
from PIL import Image
import math

import multi_inout_net

label_1 = ["male", "female"]
label_2 = ["smile", "non-smile"]

class Preprocess:
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
            img = np.array(img, dtype=np.uint8)
            img = img.transpose(2, 0, 1)  # ch, height, width
            
            img_float = np.zeros((img.shape[0],img.shape[1],img.shape[2]), dtype=np.float32)
            
            np_hist_R = np.zeros(256)
            np_hist_G = np.zeros(256)
            np_hist_B = np.zeros(256)
            
            num_pixel = img.shape[1]*img.shape[2]
            
            self.make_hist(img, np_hist_R, np_hist_G, np_hist_B)
            
            hist_R.append(np_hist_R/num_pixel)
            hist_G.append(np_hist_G/num_pixel)
            hist_B.append(np_hist_B/num_pixel)
            
            for ch in range(3):
                img_float[ch] = (img[ch]/255 - mean_rgb[ch])/std_rgb[ch]
                
            img_data.append(img_float)
            
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
    def get_batch_tensor(self):
        
        return torch.from_numpy(self.img_data.astype(np.float32)).clone(), torch.from_numpy(self.hist_R.astype(np.float32)).clone(),\
            torch.from_numpy(self.hist_G.astype(np.float32)).clone(), torch.from_numpy(self.hist_B.astype(np.float32)).clone()
            
    def make_hist(self, img, hist_r, hist_g, hist_b):
        
        for ch in range(img.shape[0]):
            for y in range(img.shape[1]):
                for x in range(img.shape[2]):
                    if ch == 0:
                        hist_r[img[ch,y,x]] += 1
                    elif ch == 1:
                        hist_g[img[ch,y,x]] += 1
                    else:
                        hist_b[img[ch,y,x]] += 1

def get_class_prob(out):
    fval = 0
    prob = 0
    
    for i in range(out.shape[1]):
        fval = abs(1/(1+math.exp(out[0][i]*(-1))))
        
        if fval > prob:
            class_no = i
            prob = fval
    return class_no, prob


    

resnet = models.resnet50()
model = multi_inout_net.Net(resnet)

model.load_state_dict(torch.load('model\\multi_inout_weight.pth', map_location='cpu'))


# modelをGPU or CPU に送る
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('--- device: ' + str(device) + ' ---')

model = model.to(device)
model.eval()

img_name_list = []
img_name_list.append('test.jpg')

# インプットデータ作成
train_data_set = Preprocess(img_name_list)
image, h1, h2, h3 = train_data_set.get_batch_tensor()

# インプットデータをdeviceへ送る
image = image.to(device)
h1 = h1.to(device)
h2 = h2.to(device)
h3 = h3.to(device)

print('')
print('--- start inference ---')
y0_out, y1_out = model(image, h1, h2, h3)
print('--- end inference ---')



y0_out = y0_out.to('cpu').detach().numpy().copy()

y1_out = y1_out.to('cpu').detach().numpy().copy()

#y0_outの1番目の要素:maleのスコア, y0_outの2番目の要素:femaleのスコア
print('y0_out', y0_out)

#y1_outの1番目の要素:smileのスコア, y1_outの2番目の要素:non smileのスコア
print('y1_out', y1_out)


y0_class_no, y0_prob =  get_class_prob(y0_out)
y1_class_no, y1_prob =  get_class_prob(y1_out)

print('y0_class_no', y0_class_no, ',', label_1[y0_class_no])
print('y0_prob', y0_prob)

print('y1_class_no', y1_class_no, ',',label_2[y1_class_no])
print('y1_prob', y1_prob)
