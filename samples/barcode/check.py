from PIL import Image
import glob
import os
from tqdm import tqdm

dataset_dir="/home/ec2-user/SageMaker/benchmarks/high_resolution"
for i in tqdm(range(80000)):
    filename = "roi_train_im/roi{}.png".format(i)
    image_path = os.path.join(dataset_dir, filename)
    im = Image.open(image_path)
    height, width = im.size
    mask_pattern = "roi_train_masks/roi_mask{}_*.jpg".format(i)
    all_masks_names = glob.glob(os.path.join(dataset_dir, mask_pattern))

    for filename in all_masks_names:    
        mask = Image.open(filename)
        h, w = mask.size
        if h != height or w != width:
            print(filename)
