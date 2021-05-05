#!/usr/bin/env python

python train_diff_block.py -pn="diff-block-f6" -name="diff_1000"
python video_compress_fusion.py -name="diff_1000" >> ./videos/diff_1000/test/bpp.txt
python video_compute.py -p="./videos/diff_1000/test/*.avi" >> ./videos/diff_1000/test/score.txt
 