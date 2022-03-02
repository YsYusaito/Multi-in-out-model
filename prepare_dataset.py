# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:04:18 2022

@author: 10087826
"""

import os
from os.path import join

from pathlib import Path
from PIL import Image

os.chdir('C:\\Users\\10087826\\Documents\\multi_in_out_model')

# outputディレクトリの指定
output_dir = f'CelebA_dataset\\'


# フォルダをリストで作成
pass_list = [f'00_male_smiling\\',
             f'01_male_Nonsmiling\\',
             f'10_female_smiling\\',
             f'11_female_Nonsmiling\\']

max_image_num_per_class = 6000
image_count_per_class = [0,0,0,0]

for path in pass_list:
    path_train_data = join(output_dir, path)
    if not os.path.exists(path_train_data):
        os.makedirs(path_train_data)

count = 0

with open("CelebA_dataset/list_attr_celeba.txt","r") as f:    ### 属性ファイルを開く
      for i in range(202599+2):   # 全部で202,599枚処理する
          line = f.readline()   # 1行データ読み込み
          line = line.split()   # データを分割
          count = count+1
          # print(count)

          if count >= 2:
              
              # male_smiling
              if line[21]=="1" and line[32]=="1":
                  image_count_per_class[0] = image_count_per_class[0] + 1
                  if image_count_per_class[0] <= max_image_num_per_class:
                      image = Image.open("img_align_celeba/"+line[0])
                      image.save(output_dir + pass_list[0] + line[0])
                  
              # male_Nonsmiling
              elif line[21]=="1" and line[32]=="-1":
                  image_count_per_class[1] = image_count_per_class[1] + 1
                  if image_count_per_class[1] <= max_image_num_per_class:
                      image = Image.open("img_align_celeba/"+line[0])
                      image.save(output_dir + pass_list[1] + line[0])
              
              # female_smiling
              if line[21]=="-1" and line[32]=="1":
                  image_count_per_class[2] = image_count_per_class[2] + 1
                  if image_count_per_class[2] <= max_image_num_per_class:
                      image = Image.open("img_align_celeba/"+line[0])
                      image.save(output_dir + pass_list[2] + line[0])
    
              # female_Nonsmiling
              elif line[21]=="-1" and line[32]=="-1":
                  image_count_per_class[3] = image_count_per_class[3] + 1
                  if image_count_per_class[3] <= max_image_num_per_class:
                      image = Image.open("img_align_celeba/"+line[0])
                      image.save(output_dir + pass_list[3] + line[0])
    
                  
# 学習・テストデータ画像への相対パスを作る

# データの分割
train_data_list = []
test_data_list = []


for path in pass_list:
    path_dir_data = Path(output_dir+path)
    list_path_img = sorted(list(path_dir_data.glob('*.jpg')))
    
    count = 0
    division_point = int(len(list_path_img)*0.9) # 学習データを90%、評価データを10%
    
    for path_img in list_path_img:
        count = count + 1
        if count < division_point:               # division_pointより小さいときは学習データに割り振る。
            train_data_list.append(str(path_img))
        else:                                    # division_pointより大きいときは評価データに割り振る。
            test_data_list.append(str(path_img))

# データ数の確認
print('---len(data_train)---')
print(len(train_data_list))

print('---len(data_test)---')
print(len(test_data_list))
            
train_data_list_file = "train_data_list.txt"
test_data_list_file = "test_data_list.txt"


train_data_list_for_output = "\n".join(train_data_list)
test_data_list_for_output = "\n".join(test_data_list)

with open(train_data_list_file, 'w') as f_train:
    f_train.write(train_data_list_for_output)
    
with open(test_data_list_file, 'w') as f_test:
    f_test.write(test_data_list_for_output)
