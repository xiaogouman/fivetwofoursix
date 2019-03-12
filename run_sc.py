# python3.5 run_sc.py <test_file_path> <model_file_path> <output_file_path>

import os
import math
import sys
import torch

def test_model(test_text_file, model_file, out_file):
    # write your code here. You can add functions as well.
		# use torch library to load model_file
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_text_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    test_model(test_text_file, model_file, out_file)
