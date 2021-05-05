#!/usr/bin/env python

# python train_diff_block.py -pn="diff-block-f6" -name="diff_1000"
# python video_compress_fusion.py -name="diff_1000" >> ./videos/diff_1000/test/bpp.txt
# python video_compute.py -p="./videos/diff_1000/test/*.avi" >> ./videos/diff_1000/test/score.txt
 
#python train_diff_block.py -name="block-f6" -pn="diff-f3"
python train_bidir.py -name="bidir-diff" -pn="bidir-base"
python video_compress_bidir.py -name="bidir-diff" > ./videos/bidir-diff/test/bpp.txt
#python video_compress_fusion.py -name="block-f6" > ./videos/block-f6/test/bpp.txt
python video_compute.py -p="./videos/bidir-diff/test/*.avi" > ./videos/bidir-diff/test/score.txt
