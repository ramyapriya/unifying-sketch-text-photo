import argparse

parser = argparse.ArgumentParser(description='Scene Sketch Text')

parser.add_argument('--exp_name', type=str, default='default', help='set experiment name')

# ----------------------------
# Dataloader Options
# ----------------------------

# For Our Dataset:
parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/phd2/dataset_paper/scene-sketch-text/data',
	help='Enter root directory of OurScene Dataset')

# For SketchyScene Dataset:
# parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/datasets/sketchyscene/SketchyScene-7k',
# 	help='Enter root directory of SketchyScene Dataset')

# For SketchyCOCO Dataset:
# parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/datasets/SketchyCOCO/Scene/',
# 	help='Enter root directory of SketchyCOCO Dataset')

# For Sketchy (object sketch) Dataset:
# parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/datasets/sbir_dataset/sketchy',
# 	help='Enter root directory of Sketchy dataset')

parser.add_argument('--max_len', type=int, default=224, help='Max Edge length of images')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--workers', type=int, default=12, help='Num of workers in dataloader')
parser.add_argument('--use_coco', action='store_true', default=False, help='use COCO captions')
parser.add_argument('--combine_type', type=str, default='concat', help='method to combine sketch+text')

opts = parser.parse_args()
