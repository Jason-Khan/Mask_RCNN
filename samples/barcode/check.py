from PIL import Image
import glob
import os
from tqdm import tqdm
import multiprocessing as mp

dataset_dir="/home/ec2-user/SageMaker/benchmarks/high_resolution"
def clean(start, end):
    for i in tqdm(range(start, end)):
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
                os.remove(filename)
                print(filename)

if __name__ == '__main__':
    num_processes = 20
    step = 8000
    processes = []
    for i in range(num_processes):
        processes.append(mp.Process(target=clean, args=(i*step, (i+1)*step)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done")