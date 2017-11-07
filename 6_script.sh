#!/bin/bash

python3 0_divide_ed_estrae_roi.py
~/Fiji.app/ImageJ-linux64 -macro 4_set_measure.ijm
python3 5_join_results.py
