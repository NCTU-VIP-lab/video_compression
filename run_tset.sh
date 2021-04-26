#!/usr/bin/env python

python train_diff_block.py -name="block-f6" -pn="diff-f3"
python video_compress_fusion.py -name="block-f6" > ./videos/block-f6/test/bpp.txt
python video_compute.py -p="./videos/block-f6/test/*.avi" > ./videos/block-f6/test/score.txt
 