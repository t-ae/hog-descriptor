import sys
import argparse
import json
import numpy as np
import skimage.feature

def main():
    parser = argparse.ArgumentParser(description="output test values")
    parser.add_argument("--image-size", nargs=2, type=int)
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--pixels-per-cell", nargs=2, type=int)
    parser.add_argument("--cells-per-block", nargs=2, type=int)
    parser.add_argument("--block-norm", type=str, default="L1")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    image = np.arange(args.image_size[0]*args.image_size[1])
    image = np.sin(image)
    image = np.abs(image)
    image = image.reshape([args.image_size[1], args.image_size[0]])
    
    if args.debug:
        imv = np.zeros_like(image)
        imh = np.zeros_like(image)
        imv[1:-1, :] = image[2:, :] - image[:-2, :]
        imh[:, 1:-1] = image[:, 2:] - image[:, :-2]
        grad = np.arctan2(imv, imh) * 180 / np.pi
        
        print("grad", grad)
        
        magnitude = np.hypot(imv, imh)
        print("magnitude", magnitude)
        
       
        f, im = skimage.feature.hog(image, 
                                    orientations=args.orientations,
                                    pixels_per_cell=args.pixels_per_cell, 
                                    cells_per_block=args.cells_per_block, 
                                   block_norm=args.block_norm,
                                   visualize=True)
        print(im)
        exit(0)

    f = skimage.feature.hog(image, 
                            orientations=args.orientations,
                            pixels_per_cell=args.pixels_per_cell, 
                            cells_per_block=args.cells_per_block, 
                            block_norm=args.block_norm)
    print(json.dumps(list(f)))
    print("size_{}_{}_ori_{}_ppc_{}_{}_bpc_{}_{}_{}".format(args.image_size[0], 
        args.image_size[1], 
        args.orientations,
        args.pixels_per_cell[0],
        args.pixels_per_cell[1],
        args.cells_per_block[0],
        args.cells_per_block[1],
        args.block_norm))

if __name__ == "__main__":
    main()
