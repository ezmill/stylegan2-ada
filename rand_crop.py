import os
import glob
import random
from PIL import Image
# files = glob.glob(os.path.join('*.png, *.jpg')
# files = os.listdir()
from os.path import join
from glob import glob

out_dir = "A:/crops_512"
files = []
for ext in ('*.png', '*.jpg'):
   files.extend(glob(join("A:/Photography/**/", ext)))

dx = dy = 512

num = 0
for file in files:
        # print(file)
        im = Image.open(file)

        new_filename = os.path.join(out_dir, str(num) + '_0.png')
        

        random.seed(4098340983)
        w, h = im.size;
        x = random.randint(0, w-dx-1)
        y = random.randint(0, h-dy-1)
    
        im.crop((x,y, x+dx, y+dy)).save(new_filename, format="png")
        num += 1