#!/usr/bin/env python

python video_compress_fusion.py -name="diff-f6" > ./videos/diff-f6/test/bpp.txt
python video_compute.py -p="./videos/diff-f6/test/*.avi" > ./videos/diff-f6/test/score.txt
 