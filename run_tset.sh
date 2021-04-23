#!/usr/bin/env python

python train_diff_block.py -name="diff-block-f6-1000" -pn="diff-f3"
python video_compress_fusion.py -name="diff-block-f6-1000" > ./videos/diff-block-f6-1000/test/bpp.txt
python video_compute.py -p="./videos/diff-block-f6-1000/test/*.avi" > ./videos/diff-block-f6-1000/test/score.txt
 