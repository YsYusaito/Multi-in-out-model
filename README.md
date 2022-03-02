# Multi in out model

# 概要
[環境]
Anacondaで仮想環境を作成し、以下のパッケージをインストールした。
使用GPU:Quadro RTX 6000  
・python 3.7.11  
・pytorch 1.6.0  
・torchvision 0.7.0  
・cuda tool kit 10.1  
・numpy 1.21.2  
・matplotlib 3.5.0  
・scikit-learn 1.0.2  
・pillow 8.4.0  
・tqdm 4.62.3  

python, pytorch, torchvision, cuda toolkit以外は最新版のものをインストールした結果、上記のようなバージョンとなった。
※tqdmのみ、pipでインストール。

※使用するGPUによって、対応するcuda tool kitや、pytorch、torch visionのバージョンが異なる。[こちら](https://pytorch.org/get-started/previous-versions/)を参照にcuda tool kitや、pytorch、torch visionの適切なバージョンをインストールすること。

[学習データ]  
・[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
・[CelebAとは](http://cedro3.com/ai/celeba-dataset-attribute/)

![celeba_overview](https://user-images.githubusercontent.com/80863816/156318009-733d02b1-4027-4bcb-9438-a6607b34e0c5.PNG)  
↑のように、笑っている男女の画像が30万枚近くある。

[ネットワーク構造]  
![multi_inout_network](https://user-images.githubusercontent.com/80863816/156318015-6f4096a6-8dfc-4809-9c08-2236d7123adb.PNG)   
[ネットワークの元ネタ](https://dajiro.com/entry/2020/06/27/160255)

・学習では、画像が、male, femaleのどちらかなのか(第一のラベル)、smiling, non smilingのどちらなのか(第二のラベル)を判別できるようにした。
・モデル全体のパラメータを学習により、更新した。

・male, female/smiling, non smiling それぞれのくくりに対して6000枚の画像、計24000枚をデータセットから取得した。(メモリの都合や、学習時間を抑えるためにすべての画像は用いなかった。)

・学習データの枚数：テストデータの枚数 = 9:1 の比率とした  
・epoch数：3  
・ラーニングレート：0.0001

[学習時のロス・精度]   
・ロス     
![loss](https://user-images.githubusercontent.com/80863816/156318018-d3698a64-7fc8-48be-9e48-ede1326cbcb1.png)   
・精度   
![acc](/uploads/b78dd896259049d6db0ce6c7cafcd325/acc.png)   

# モデル学習までの流れ
1. git clone  
   リンク：https://enosta.olympus.co.jp/gitlab_01/ai-community/model-zoo/multi-in-out-model.git
2. 学習データダウンロード
   - celebaの[公式サイト](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)にアクセス  
   - Google Driveをクリック  
   ![celeba_google_drive](/uploads/0e9d642230c17ca4b305500f246878c1/celeba_google_drive.PNG)  
   - imgをクリック  
   ![celeba_img](/uploads/d97ddaa3226180a110a1cf9bda5ed18f/celeba_img.PNG)  
   - img_align_celeba.zipをダウンロード  
   ![celeba_zip](/uploads/9147292c88e282b69b21b89d16218b6e/celeba_zip.PNG)  
   - Annoをクリック  
   ![celeba_anno](/uploads/1836a10576ab606e11116f5f8f45acfc/celeba_anno.PNG)  
   - list_attr_celeba.txtのダウンロード  
   ![celeba_list_attr](/uploads/092ad1fc3807d21f0c4122e70e5dd15b/celeba_list_attr.PNG)

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
   ![inference_log](/uploads/12f554738f860fb88eab8505c585c67f/inference_log.PNG)
   
   y0_outの1つ目の要素： maleのスコア   
   y0_outの2つ目の要素： femaleのスコア   

   y1_outの1つ目の要素： smilingのスコア   
   y1_outの2つ目の要素： non-smilingのスコア   



