import argparse

parser = argparse.ArgumentParser(description='Unifying Sketch, Text and Photo')

# ----------------------------
# Dataloader Options
# ----------------------------

# For SketchyCoco:
# ------------------

parser.add_argument('--root_dir', type=str, default='/vol/research/sketchcaption/ramya/dissertation_exps/data',
	help='Enter root directory of CustomSketchyCOCO Dataset')
parser.add_argument('--split', type=str, default='split1',
	help='Enter split of dataset')
parser.add_argument('--log_dir', type=str, default='/vol/research/sketchcaption/ramya/dissertation_exps/training_runs/logs_july12th_v2/')
parser.add_argument('--max_len', type=int, default=224, help='Max Edge length of images')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--workers', type=int, default=12, help='Num of workers in dataloader')
parser.add_argument('--combine_type', type=str, default='additive', help='method to combine sketch+text')
parser.add_argument('--config_file', type=str, default='params.yaml', help='Configuration params')

opts = parser.parse_args()
