import torch
import argparse, textwrap
from pathlib import Path
import numpy as np
from PIL import Image



def RGB2LMS(img_file_path, if_convert=True):
    image = Image.open(img_file_path)
    rgb = np.asarray(image)
    rgb2xyz = np.array([[0.4124,0.3756,0.1805],[0.2126,0.7152,0.0722],[0.0192,0.1192,0.9505]])
    xyz2xyzd = np.array([[0.1884,0.6597,0.1016],[0.2318,0.8116,-0.0290],[0,0,1]])
    xyz2rgb = np.linalg.inv(rgb2xyz)
    M1 = np.matmul(xyz2rgb,xyz2xyzd)
    M2 = np.matmul(M1,rgb2xyz)
    rgb_reshaped = rgb.reshape((rgb.shape[0]*rgb.shape[1]),3)
    rgbd_reshaped = np.matmul(rgb_reshaped,M2.T).astype(int)
    rgbd = rgbd_reshaped.reshape(rgb.shape)
    rgbd_final = np.clip(rgbd,0,255)
    rgbd_image = Image.fromarray(rgbd_final.astype(np.uint8))
    img_name = img_file_path.split("\\")[-1].split(".")[0]
    rgbd_image.save("temp" + '/' + img_name+ "_transformed" + ".jpg")


    return "temp" + '/' + img_name+ "_transformed" + ".jpg"




if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="EE 610 Project", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", help="Takes in path to input image file", type=str, metavar="i")
    parser.add_argument("--model", help=textwrap.dedent('''Enter either of
        [1] d   : DenseNet
        [2] r   : ResNet
        [3] e   : EfficientNet'''), type=str, metavar="m", default='d')
    parser.add_argument("--convert", help="Converts image to canine vision space (default is False)", type=bool, metavar="c", default=True)
    args = parser.parse_args()
    Path("temp").mkdir(exist_ok=True)
    
    if args.convert == False:
        img_path = RGB2LMS(args.image, False)
    else:
        img_path = RGB2LMS(args.image, True)

    
