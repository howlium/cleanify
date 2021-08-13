import cv2
import os
import numpy as np
from tqdm import tqdm
from io import StringIO
import argparse
import sys
import random

out_w = 64
out_h = 64

# construct the argument parser
parser = argparse.ArgumentParser(description = "Filter a clean image, output a dirty one")
parser.add_argument("-t", "--tile", action="store_true",
                    help="use tiles to produce multiple subimages (as opposed to scaling once)")
parser.add_argument("-j", "--jpeg", type=int,
                    help="JPEG-compress with quality (0-100%) and decompress")
parser.add_argument("-b", "--blur", type=int,
                    help="Gaussian blur with [odd] pixel radius")
parser.add_argument("-r", "--rblur", type=int,
                    help="radial blur with given iteration count (0-10")
parser.add_argument("-n", "--noise", type=float,
                    help="add noise (0-400%)")
parser.add_argument("-x", "--xout", action="store_true",
                    help="draw a dark red 1-pixel-thick X over the image")
parser.add_argument("-p", "--plus", action="store_true",
                    help="draw white horizontal and vertical lines at random positions")
parser.add_argument("-i", "--invert", action="store_true",
                    help="invert the image")
args = parser.parse_args()

# validate params
abort = False
if args.jpeg is not None and (args.jpeg < 0 or args.jpeg > 100):
    print("JPEG quality must be between 0 and 100")
    abort = True
if args.blur is not None and args.blur != 0 and args.blur % 2 == 0:
    print("blur radius must be odd")
    abort = True
if args.noise is not None and (args.noise < 0 or args.noise > 400):
    print("Noise % must be between 0 and 400")
    abort = True
if abort:
    sys.exit()

# Directories
# Given a bunch of larger images in the input/raw:
# (1) scale and crop to a small, square, manageable thumbnail size
# (2) save that into input/clean
# (3) add the requested perturbation
# (4) save that into input/dirty

os.makedirs('input/dirty', exist_ok=True)
os.makedirs('input/clean', exist_ok=True)
raw_dir = 'input/raw'
clean_dir = 'input/clean'
dirty_dir = 'input/dirty'

# Iterate through files in input/raw
images = os.listdir(raw_dir)
if '.DS_Store' in images:
    images.remove('.DS_Store') # For macOS: Skip invisible Desktop Services Store file.

# Dirtify function
def add_dirt(in_img, w, h):

    if args.invert:
        # Invert the image
        img = 255 - in_img

    if args.jpeg is not None:
        # Compress with JPEG at given quality, then uncompress
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg]
        result, jpeg_bytes = cv2.imencode('.jpg', in_img, encode_param)
        img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

    if args.noise is not None:
        # Mix 3 channels of simple noise over the image
        h, w, d = in_img.shape[0], in_img.shape[1], in_img.shape[2]
        noise_img = np.random.rand(h,w,d) * 255
        mask_img = np.random.rand(h,w,d)
        for y in range(h):
            for x in range(w):
                for c in range(d):
                    if args.noise <= 100:
                        alpha = mask_img[y,x,c] * (args.noise / 100.0) # Linear up to 100%
                    else:
                        alpha = mask_img[y,x,c] ** (100.0 / args.noise) # Curved above 100%
                    img[y,x,c] = alpha * noise_img[y,x,c] + (1 - alpha) * in_img[y,x,c]
        
    if args.blur is not None:
        # Apply Gaussian Blur with given pixel radius
        img = cv2.GaussianBlur(in_img, (args.blur, args.blur), 0)
        
    if args.rblur is not None:
        # Apply a radial blur
        radius = 0.01
        mid_y = h / 2
        mid_x = w / 2
        growMapx = np.abs(np.tile(np.arange(h) + ((np.arange(h) - mid_x) * radius), (w, 1)).astype(np.float32))
        shrinkMapx = np.abs(np.tile(np.arange(h) - ((np.arange(h) - mid_x) * radius), (w, 1)).astype(np.float32))
        growMapy = np.abs(np.tile(np.arange(w) + ((np.arange(w) - mid_y) * radius), (h, 1)).transpose().astype(np.float32))
        shrinkMapy = np.abs(np.tile(np.arange(w) - ((np.arange(w) - mid_y) * radius), (h, 1)).transpose().astype(np.float32))
        
        for i in range(args.rblur):
            tmp1 = cv2.remap(in_img, growMapx, growMapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            tmp2 = cv2.remap(in_img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    
    if args.plus:
        # superimpose a white plus at a random position
        x = round(random.random() * w)
        y = round(random.random() * h)
        img = in_img # make a copy so we don't overdraw the clean image
        cv2.line(img, (0,y), (w, y), (255,255,255), 1)
        cv2.line(img, (x,0), (x, h), (255,255,255), 1)
    
    if args.xout:
        # Draw a dark red X through the image
        img = in_img # make a copy so we don't overdraw the clean image
        cv2.line(img, (0,0), (w,h), (0,0,64), 2)
        cv2.line(img, (0,h), (w,0), (0,0,64), 2)
    
    return img


# Step through images
for i, clean in tqdm(enumerate(images), total=len(images)):
    filename = f"{raw_dir}/{images[i]}"
    clean = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    # resize and crop to make a 64 x 64 image
    in_h = clean.shape[0]
    in_w = clean.shape[1]
    
    if args.tile:
        raw_tile_h = 512
        raw_tile_w = 512
        for y in range(int(in_h / raw_tile_h)):
            top = y * raw_tile_h
            bottom = top + raw_tile_h
            for x in range(int(in_w / raw_tile_w)):
                left = x * raw_tile_w
                right = left + raw_tile_w
                tile = clean[top:bottom, left:right]
                
                # Scale from raw tile size to output size
                scale = out_h / raw_tile_h
                tile = cv2.resize(tile, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)

                # Add that dirt!
                dirty = add_dirt(tile, out_w, out_h)
                
                # And save files
                out_name = os.path.splitext(images[i])[0] + f" ({x},{y}).png" # Change the extension to png (for lossless image)
                cv2.imwrite(f"{clean_dir}/{out_name}", tile)
                cv2.imwrite(f"{dirty_dir}/{out_name}", dirty)

    else:
        scale = max(out_h / in_h, out_w / in_w)

        # resize and crop
        mid_y = scale * in_h / 2
        mid_x = scale * in_w / 2
        clean = cv2.resize(clean, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
        clean = clean[round(mid_y - out_h/2):round(mid_y + out_h/2), round(mid_x - out_w/2):round(mid_x + out_w/2)]

        # Add that dirt!
        dirty = add_dirt(clean, out_w, out_h)
        
        out_name = os.path.splitext(images[i])[0] + ".png" # Change the extension to png (for lossless image)
        cv2.imwrite(f"{clean_dir}/{out_name}", clean)
        cv2.imwrite(f"{dirty_dir}/{out_name}", dirty)
    
print('DONE')
