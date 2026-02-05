import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--input_video', type=str, help='the path of input video file')
        self.parser.add_argument('--garment_name', type=str, default='example_garment', help='id of the target garment')
        self.parser.add_argument('--use_beta', action='store_true', default=True, 
                                help='Use Extended Hybrid Representation (16ch input with beta). '
                                     'Set for models trained with upperbody_training_beta.py')
        self.parser.add_argument('--no_beta', action='store_true', default=False,
                                help='Disable beta (6ch input). Use for models trained with upperbody_training.py')
        self.parser.add_argument('--fix_first_frame_beta', action='store_true', default=True,
                                help='Use first frame beta throughout entire video (default: True). '
                                     'Ensures temporal consistency of body shape.')
        self.parser.add_argument('--no_fix_first_frame_beta', action='store_true', default=False,
                                help='Disable first frame beta fixing. Extract beta fresh from each frame.')
        self.parser.add_argument('--output', type=str, default='./output.mp4',
                                help='Output video path')
        self.initialized=True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


