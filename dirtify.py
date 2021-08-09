import cv2
import os
import numpy as np
from tqdm import tqdm
from io import StringIO
import argparse
import sys

out_w = 64
out_h = 64

# construct the argument parser
parser = argparse.ArgumentParser(description = "Filter a clean image, output a dirty one")
parser.add_argument("-j", "--jpeg", type=int,
                    help="JPEG-compress with quality (0-100%) and decompress")
parser.add_argument("-b", "--blur", type=int,
                    help="Gaussian blur with [odd] pixel radius")
parser.add_argument("-n", "--noise", type=float,
                    help="add noise (0-400%)")
parser.add_argument("-x", "--xout", action="store_true",
                    help="draw a dark red 1-pixel-thick X over the image")
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
# Given a bunch of larger images in the ../input/raw:
# (1) scale and crop to a small, square, manageable thumbnail size
# (2) save that into ../input/clean
# (3) add the requested perturbation
# (4) save that into ../input/dirty

os.makedirs('../input/dirty', exist_ok=True)
os.makedirs('../input/clean', exist_ok=True)
raw_dir = '../input/raw'
clean_dir = '../input/clean'
dirty_dir = '../input/dirty'

# Iterate through files in ../input/raw
images = os.listdir(raw_dir)
if '.DS_Store' in images:
    images.remove('.DS_Store') # For macOS: Skip invisible Desktop Services Store file.

for i, img in tqdm(enumerate(images), total=len(images)):
    filename = f"{raw_dir}/{images[i]}"
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    # resize and crop to make a 64 x 64 image
    in_h = img.shape[0]
    in_w = img.shape[1]
    scale = max(out_h / in_h, out_w / in_w)

    # resize...
    img = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
    mid_h = scale * in_h / 2
    mid_w = scale * in_w / 2

    # ...and crop
    img = img[round(mid_h - out_h/2):round(mid_h + out_h/2), round(mid_w - out_w/2):round(mid_w + out_w/2)]
    out_name = os.path.splitext(images[i])[0] + ".png" # Change the extension to png (for lossless image)
    cv2.imwrite(f"{clean_dir}/{out_name}", img)

    # Add that dirt!
    dirty = img
    
    if args.invert:
        # Invert the image
        dirty = 255 - dirty

    if args.jpeg is not None:
        # Compress with JPEG at given quality, then uncompress
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg]
        result, jpeg_bytes = cv2.imencode('.jpg', dirty, encode_param)
        dirty = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

    if args.noise is not None:
        # Mix 3 channels of simple noise over the image
        h, w, d = dirty.shape[0], dirty.shape[1], dirty.shape[2]
        noise_img = np.random.rand(h,w,d) * 255
        mask_img = np.random.rand(h,w,d)
        for y in range(h):
            for x in range(w):
                for c in range(d):
                    if args.noise <= 100:
                        alpha = mask_img[y,x,c] * (args.noise / 100.0) # Linear up to 100%
                    else:
                        alpha = mask_img[y,x,c] ** (100.0 / args.noise) # Curved above 100%
                    dirty[y,x,c] = alpha * noise_img[y,x,c] + (1 - alpha) * dirty[y,x,c]
        
    if args.blur is not None:
        # Apply Gaussian Blur with given pixel radius
        dirty = cv2.GaussianBlur(dirty, (args.blur, args.blur), 0)
        
    if args.xout:
        # Draw a dark red X through the image
        cv2.line(dirty, (0,0    ), (out_w,out_h), (0,0,64), 1)
        cv2.line(dirty, (0,out_h), (out_w,0    ), (0,0,64), 1)
    
    cv2.imwrite(f"{dirty_dir}/{out_name}", dirty)
    
print('DONE')
