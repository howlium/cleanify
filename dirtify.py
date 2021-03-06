import cv2
import os
import numpy as np
from tqdm import tqdm
from io import StringIO
import argparse
import sys
import random
import itertools
import shutil

out_w = 64
out_h = 64

# construct the argument parser
parser = argparse.ArgumentParser(description = "Filter a clean image, output a dirty one")
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

# find the right folder name for this effect
def append_names(str1, str2):
    if str1=='':
        return str2
    else:
        return f'{str1}-{str2}'

fxName = ''
if args.jpeg is not None:
    fxName = append_names(fxName, 'jpeg'+str(args.jpeg))
if args.blur is not None:
    fxName = append_names(fxName, 'blur'+str(args.blur))
if args.rblur:
    fxName = append_names(fxName, 'rblur')
if args.noise is not None:
    fxName = append_names(fxName, 'noise'+str(args.noise))
if args.xout:
    fxName = append_names(fxName, 'xout')
if args.plus:
    fxName = append_names(fxName, 'plus')
if args.invert:
    fxName = append_names(fxName, 'invert')
if fxName == '':
    fxName = 'null'

# define directories
clean_dir = 'input/clean'
clean_tiled_dir = clean_dir + '/tiled'
shutil.rmtree(clean_tiled_dir, ignore_errors=True)
os.makedirs(clean_tiled_dir, exist_ok=True)
clean_scaled_dir = clean_dir + '/scaled'
shutil.rmtree(clean_scaled_dir, ignore_errors=True)
os.makedirs(clean_scaled_dir, exist_ok=True)

dirty_dir = 'input/dirty'
dirty_tiled_dir = f'{dirty_dir}/{fxName}/tiled'
shutil.rmtree(dirty_tiled_dir, ignore_errors=True)
os.makedirs(dirty_tiled_dir, exist_ok=True)
dirty_scaled_dir = f'{dirty_dir}/{fxName}/scaled'
shutil.rmtree(dirty_scaled_dir, ignore_errors=True)
os.makedirs(dirty_scaled_dir, exist_ok=True)

raw_dir = 'input/raw'

# Dirtify function
def add_dirt(img):
    h,w,d = img.shape[:3]
    
    if args.invert:
        # Invert the image
        img = 255 - img

    if args.jpeg is not None:
        # Compress with JPEG at given quality, then uncompress
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg]
        result, jpeg_bytes = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

    if args.noise is not None:
        # Mix 3 channels of simple noise over the image
        noise_img = np.random.rand(h,w,d) * 255
        mask_img = np.random.rand(h,w,d)
        for y, x, c in itertools.product(range(h), range(w), range(d)):
            if args.noise <= 100:
                alpha = mask_img[y,x,c] * (args.noise / 100.0) # Linear up to 100%
            else:
                alpha = mask_img[y,x,c] ** (100.0 / args.noise) # Curved above 100%
            img[y,x,c] = alpha * noise_img[y,x,c] + (1 - alpha) * img[y,x,c]
        
    if args.blur is not None:
        # Apply Gaussian Blur with given pixel radius
        img = cv2.GaussianBlur(img, (args.blur, args.blur), 0)
        
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
            tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    
    if args.plus:
        # superimpose a white plus at a random position
        x = round(random.random() * w)
        y = round(random.random() * h)
        cv2.line(img, (0,y), (w, y), (255,255,255), 1)
        cv2.line(img, (x,0), (x, h), (255,255,255), 1)
    
    if args.xout:
        # Draw a dark red X through the image
        cv2.line(img, (0,0), (w,h), (0,0,64), 1)
        cv2.line(img, (0,h), (w,0), (0,0,64), 1)
    
    return img


# Iterate through raw images
images = os.listdir(raw_dir)
if '.DS_Store' in images:
    images.remove('.DS_Store') # For macOS: Skip invisible Desktop Services Store file.

for i, clean in tqdm(enumerate(images), total=len(images)):
    filename = f"{raw_dir}/{images[i]}"
    raw_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    in_h, in_w = raw_img.shape[:2]
    out_name_base = os.path.splitext(images[i])[0]
    
    # First do the scaled set. Resize each raw image and crop to fill a 64 x 64 image
    scale = max(out_h / in_h, out_w / in_w)
    mid_y = scale * in_h / 2
    mid_x = scale * in_w / 2

    clean_img = cv2.resize(raw_img, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
    out_img = clean_img[round(mid_y - out_h/2):round(mid_y + out_h/2), round(mid_x - out_w/2):round(mid_x + out_w/2)]
    out_name = out_name_base + ".png" # Change the extension to png (for lossless image)
    cv2.imwrite(f"{clean_scaled_dir}/{out_name}", out_img)
    # Add that dirt!
    out_img = add_dirt(out_img)
    cv2.imwrite(f"{dirty_scaled_dir}/{out_name}", out_img)
    
    # Now do the tiled set: export 2x3 or 3x2 512x512-pixel tiles from each 1536x2048 or 2048x1536 raw image
    raw_tile_h = 512
    raw_tile_w = 512
    for y, x in itertools.product(range(in_h // raw_tile_h), range(in_w // raw_tile_w)):
        top = y * raw_tile_h
        bottom = top + raw_tile_h
        left = x * raw_tile_w
        right = left + raw_tile_w
        tile = raw_img[top:bottom, left:right]
        # Scale from raw tile size to output size
        scale = out_h / raw_tile_h
        
        tile = cv2.resize(tile, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
        out_name = out_name_base + f" ({x},{y}).png" # Change the extension to png (for lossless image)
        cv2.imwrite(f"{clean_tiled_dir}/{out_name}", tile)
        # Add that dirt!
        out_img = add_dirt(tile)
        cv2.imwrite(f"{dirty_tiled_dir}/{out_name}", out_img)

print('DONE')
