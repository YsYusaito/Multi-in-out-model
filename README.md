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
・[about CelebA](http://cedro3.com/ai/celeba-dataset-attribute/)

![celeba_overview](https://user-images.githubusercontent.com/80863816/156318009-733d02b1-4027-4bcb-9438-a6607b34e0c5.PNG)  
↑There are nearly 300,000 images of men and women laughing.

[Network]  
![multi_inout_network](https://user-images.githubusercontent.com/80863816/156318015-6f4096a6-8dfc-4809-9c08-2236d7123adb.PNG)   
[Original information of this network](https://dajiro.com/entry/2020/06/27/160255)

・In the learning process, we tried to determine whether the image was male or female (first label), and whether it was smiling or non-smiling (second label).
・The parameters of the entire model were updated by learning.

・6000 images were obtained from the dataset for each of male, female/smiling, and non-smiling, for a total of 24000 images. (We did not use all the images to save memory and training time.)

・number of training data：number of test data = 9:1  
・epoch：3  
・learning rate：0.0001

[Training loss・accuracy]   
・loss     
![loss](https://user-images.githubusercontent.com/80863816/156318018-d3698a64-7fc8-48be-9e48-ede1326cbcb1.png)   
・accuracy   
![acc](https://user-images.githubusercontent.com/80863816/156318005-aff1b6ad-b3be-4252-b1b1-0f0cb54d3a7b.png) 

# Procedure of model training
1. git clone  
   Link：git@github.com:YsYusaito/Multi-in-out-model.git or 
2. down load data
   - access to [homepage of CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   - click Google Drive  
   ![celeba_google_drive](https://user-images.githubusercontent.com/80863816/156318951-86513de3-fa6f-4578-b515-d9c92625680c.PNG)  
   - click img  
   ![celeba_img](https://user-images.githubusercontent.com/80863816/156318958-99702ed0-1aaa-4600-9864-90c1fae2db24.PNG)  
   - download img_align_celeba.zip  
   ![celeba_zip](https://user-images.githubusercontent.com/80863816/156318966-eeb01ce5-158b-41e8-8b1a-51ee19705562.PNG)  
   - click Anno  
   ![celeba_anno](https://user-images.githubusercontent.com/80863816/156318967-0067d57d-1117-4f8e-91e9-2582a729e7cf.PNG)  
   - download list_attr_celeba.txt  
   ![celeba_list_attr](https://user-images.githubusercontent.com/80863816/156318963-c0172936-8a24-4f1b-8ae5-ef4ffc2e0d86.PNG)

3. In the folder where this readme is located, extract img_align_celeba.zip. Then you will find the img_align_celeba folder.
4. Create a folder "CelebA_dataset" in the folder where this readme is located.
5. Place list_attr_celeba.txt directly under the folder "CelebA_dataset".
6. Execute preprare_dataset.py
   The execution of this script creates folders "00_male_smiling", "01_male_Nonsmiling", "10_female_smiling", and "11_female_Nonsmiling" in the folder "CelebA_dataset", which  contains images for each class.
7. Execute train_multi_inout_model.py 
   - model will be output in the folder "model".  
   　※multi_inout_weight.pth → torch.save(model.state_dict(), 'Destination path and save name')  
       multi_inout_model.pth → torch.save(model, 'Destination path and save name')  
       [about saving models by pytorch](https://takaherox.hatenablog.com/entry/2021/01/09/230332)


   # Final folder structure (the folder where readme is stored)

   ・CelebA_dataset  
     00_male_smiling  
     01_male_Nonsmiling  
     10_female_smiling  
     11_female_Nonsmiling  

   ・img_align_celeba   
   
   ・model

   # Inference
   Execute test_inference.py   
   If the following log is displayed, success.  
   ![inference_log](https://user-images.githubusercontent.com/80863816/156318013-a4c16837-5f0b-4d1c-be59-6c96d2b50c24.PNG)  
   
   First element of y0_out： score of male   
   Second element of y0_out： score of female   

   First element of y1_out： score of smiling   
   Second element of y1_out： score of non-smiling   



