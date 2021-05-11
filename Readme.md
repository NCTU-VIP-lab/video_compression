# Video compression

[Project hackmd](https://hackmd.io/8lFXE2SpQZ-CpB3nm8ldjw?both)

## download
[data&weight](https://drive.google.com/file/d/1ng0wLzDqTljfryn9F4A0GCWSp7LYeQ4w/view?usp=sharing)

[train data](https://drive.google.com/file/d/1GOS2ukeyHUU5Z9M7d3PxJAAHsNY2IEOU/view?usp=sharing)

## File Structure

```
.
├── bpg
├── checkpoints
│   └── diff-block-f6(base)
├── pytorch_spynet
├── UVG_1080
│   └── testing file...
├── vimeo
│   ├── AoT.h5
│   └── clip(the second download file)
├── xxx.py
└── xxx.py

```

## Environment Required

```
torch 1.8.1
cuda 10.2 or 11.1
compressai
```
其他就他要裝甚麼就裝甚麼

## bpg codec

### for windows
data&weight 裡面就有了
video_compress_fusion.py要選".\\"開頭的

### for ubunte
```
sudo apt-get install -y  libpng-dev
sudo apt-get install -y  libjpeg8-dev
sudo apt-get install -y  yasm
sudo apt-get install -y  libsdl1.2-dev
sudo apt-get install -y  libsdl-image1.2-dev

mkdir bpg
cd bpg
wget http://bellard.org/bpg/libbpg-0.9.8.tar.gz
tar xzf libbpg-0.9.8.tar.gz
cd libbpg-0.9.8/

vim Makefile
"
#USE_X265=y
USE_JCTVC=y
"
make -j 8
sudo make install
```
## Parameters
```
lamda 200, 1000, 2000, 4000
Q = 37, 32, 27, 22 for ubuntu
Q = 39, 34, 29, 24 for win10
```

要改的地方

./config.py
```python
class config_train(object):    
    batch_size = 2    
    lambda_X = 2000
    use_flow_residual = True
    use_block = True

class config_test(object):
    lambda_X = 2000
    use_flow_residual = True
    use_block = True

```

./video_compress_fusion.py

line 115
```python
I_QP =  29 
```

## train and testing
ubuntu 可用script windows就一行一行跑
### windows

```python

python train_diff_block.py -name="diff_block_200" -pn="diff-block-f6" 
python video_compress_fusion.py -name="diff_block_200" >> ./videos/diff_block_200/test/bpp.txt
python video_compute.py -p="./videos/diff_block_200/test/*.avi" >> ./videos/diff_block_200/test/score.txt
 
```

### ubuntu

```python
#!/usr/bin/env python

python train_diff_block.py -name="diff_block_200" -pn="diff-block-f6"
python video_compress_fusion.py -name="diff_block_200" > ./videos/diff_block_200/test/bpp.txt
python video_compute.py -p="./videos/diff_block_200/test/*.avi" > ./videos/diff_block_200/test/score.txt

```
跑完將bpp.txt, score.txt 放到 [here](https://drive.google.com/drive/folders/1wUrI00bep61zsHdecPK4PiHHHxJPFRlN?usp=sharing)
 
開資料夾寫一下是哪個方法, lamda多少

thanks~



















