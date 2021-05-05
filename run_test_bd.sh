#!/usr/bin/env python

#python video_compress_bidir.py -name="bidir-diff" > ./videos/bidir-diff/test/bpp.txt
#python video_compute.py -p="./videos/bidir-diff/test/*.avi" > ./videos/bidir-diff/test/score.txt

#python train_bidir_block.py -name="bidir-bd" -pn="bidir-bd"
python video_compress_bidir_bd.py -name="bidir-diff" > ./videos/bidir-diff/test/bpp.txt
python video_compute.py -p="./videos/bidir-diff/test/*.avi" > ./videos/bidir-diff/test/score.txt
