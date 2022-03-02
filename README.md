# Multi in out model

# Overview
[Environment]
I created a virtual environment with Anaconda and installed the following packages.
GPU:Quadro RTX 6000  
・python 3.7.11  
・pytorch 1.6.0  
・torchvision 0.7.0  
・cuda tool kit 10.1  
・numpy 1.21.2  
・matplotlib 3.5.0  
・scikit-learn 1.0.2  
・pillow 8.4.0  
・tqdm 4.62.3  

I installed the latest versions of everything except python, pytorch, torchvision, and cuda toolkit.
※Only tqdm was installed by pip.

※Supported cuda tool kit, pytorch, and torch vision versions differ depending on the GPU used.   
Please install the appropriate versions of the cuda tool kit, pytorch, and torch vision, see [here](https://pytorch.org/get-started/previous-versions/).

[Data]  
・[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
・[CelebAとは](http://cedro3.com/ai/celeba-dataset-attribute/)

![celeba_overview](https://user-images.githubusercontent.com/80863816/156318009-733d02b1-4027-4bcb-9438-a6607b34e0c5.PNG)  
↑There are nearly 300,000 images of men and women laughing.

[Network]  
![multi_inout_network](https://user-images.githubusercontent.com/80863816/156318015-6f4096a6-8dfc-4809-9c08-2236d7123adb.PNG)   
[Original information of this network](https://dajiro.com/entry/2020/06/27/160255)

・In the learning process, we tried to determine whether the image was male or female (first label), and whether it was smiling or non-smiling (second label).
・The parameters of the entire model were updated by learning.

・6000 images were obtained from the dataset for each of male, female/smiling, and non-smiling, for a total of 24000 images. (We did not use all the images to save memory and training time.)

・学習データの枚数：テストデータの枚数 = 9:1 の比率とした  
・epoch数：3  
・ラーニングレート：0.0001

[学習時のロス・精度]   
・ロス     
![loss](https://user-images.githubusercontent.com/80863816/156318018-d3698a64-7fc8-48be-9e48-ede1326cbcb1.png)   
・精度   
![acc](https://user-images.githubusercontent.com/80863816/156318005-aff1b6ad-b3be-4252-b1b1-0f0cb54d3a7b.png) 

# モデル学習までの流れ
1. git clone  
   リンク：https://enosta.olympus.co.jp/gitlab_01/ai-community/model-zoo/multi-in-out-model.git
2. 学習データダウンロード
   - celebaの[公式サイト](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)にアクセス  
   - Google Driveをクリック  
   ![celeba_google_drive](https://user-images.githubusercontent.com/80863816/156318951-86513de3-fa6f-4578-b515-d9c92625680c.PNG)  
   - imgをクリック  
   ![celeba_img](https://user-images.githubusercontent.com/80863816/156318958-99702ed0-1aaa-4600-9864-90c1fae2db24.PNG)  
   - img_align_celeba.zipをダウンロード  
   ![celeba_zip](https://user-images.githubusercontent.com/80863816/156318966-eeb01ce5-158b-41e8-8b1a-51ee19705562.PNG)  
   - Annoをクリック  
   ![celeba_anno](https://user-images.githubusercontent.com/80863816/156318967-0067d57d-1117-4f8e-91e9-2582a729e7cf.PNG)  
   - list_attr_celeba.txtのダウンロード  
   ![celeba_list_attr](https://user-images.githubusercontent.com/80863816/156318963-c0172936-8a24-4f1b-8ae5-ef4ffc2e0d86.PNG)

3. 本readmeがあるフォルダ内で、img_align_celeba.zipを展開。するとimg_align_celebaフォルダが出くる。
4. 本readmeがあるフォルダ内で、フォルダ「CelebA_dataset」を作成
5. フォルダ「CelebA_dataset」直下にlist_attr_celeba.txtを配置
6. preprare_dataset.pyの実行
   このスクリプトの実行によって、フォルダ「CelebA_dataset」にフォルダ「00_male_smiling」「01_male_Nonsmiling」「10_female_smiling」
   「11_female_Nonsmiling」が作成され、クラスごとの画像が入っている。
7. train_multi_inout_model.pyの実行  
   - フォルダ「model」内にモデルが出力される。  
   　※multi_inout_weight.pth → torch.save(model.state_dict(), '保存先のパス・保存名')で保存している。  
       multi_inout_model.pth → torch.save(model, '保存先のパス・保存名')で保存で保存している。  
       [pytorchモデル保存について](https://takaherox.hatenablog.com/entry/2021/01/09/230332)

※スクリプトはコマンドでもIDEでもどちらで実行してもOK   

   # 最終的なフォルダ構成(readmeが保存されているフォルダ)

   ・CelebA_dataset  
     00_male_smiling  
     01_male_Nonsmiling  
     10_female_smiling  
     11_female_Nonsmiling  

   ・img_align_celeba   
   
   ・model

   # モデルを用いた推論
   test_inference.pyの実行   
   以下のログが表示されたら、成功   
   ![inference_log](https://user-images.githubusercontent.com/80863816/156318013-a4c16837-5f0b-4d1c-be59-6c96d2b50c24.PNG)
   
   y0_outの1つ目の要素： maleのスコア   
   y0_outの2つ目の要素： femaleのスコア   

   y1_outの1つ目の要素： smilingのスコア   
   y1_outの2つ目の要素： non-smilingのスコア   



