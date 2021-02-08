import cv2 as cv
import numpy as np
import yaml
from pathlib import Path
import sys
import os
import numpy
import glob
from mot_io import MotGt


font                   = cv.FONT_HERSHEY_COMPLEX
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 1
shadowColor = (0,0,0)

def fit_coordinate_in_box(coord, lower_bound, upper_bound):
    coord = lower_bound if coord < lower_bound else coord
    coord = upper_bound if coord > upper_bound else coord
    return coord

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Accuracy of TDOT model on a specific video..')

    parser.add_argument('--image-folder', metavar='IF', type=str,
                        default=None,
                        help='Enter folder path of the frames')

    parser.add_argument('--output-folder', metavar='OF', type=str,
                        default=None,
                        help='Enter output folder paths to save bbox drawn images')

    parser.add_argument('--gt-file', metavar='GTF', type=str,
                        default=None,
                        help='Enter the full path of the MOT ground-truth file')


    args = parser.parse_args()


    image_folder = Path(args.image_folder)
    out_dir = Path(args.output_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    gt_file_path = Path(args.gt_file)

    mot_gt = MotGt(gt_file_path.__str__())

    frame_counter = 0

    for image_filename in os.listdir(image_folder.__str__()):
        # if frame_counter == 0:
        #     frame_counter += 1
        #     continue
        frame_id = int(image_filename.split('.')[0])
        print("Filename: {}".format(image_filename))
        img = cv.imread(os.path.join(image_folder.__str__(), image_filename))
        dims = img.shape
        frame_height, frame_width = dims[0], dims[1]

        targets_to_draw = mot_gt.get_targets_in_frame(frame_id)

        for target in targets_to_draw:
            _, bbox, visibility = target.get_state_in_frame(frame_id)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            x1 = fit_coordinate_in_box(x1, 0, frame_width)
            x2 = fit_coordinate_in_box(x2, 0, frame_width)
            y1 = fit_coordinate_in_box(y1, 0, frame_height)
            y2 = fit_coordinate_in_box(y2, 0, frame_height)

            box_color = (0, 0, 255) if target.activity == 0 else (0, 255, 0)
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 1)

            cv.putText(img, "t:{}, v:{:4.3f}".format(target.type, visibility) ,
                       (x1, y1),
                       font,
                       fontScale,
                       shadowColor,
                       lineType)
            cv.putText(img, "t:{}, v:{:4.3f}".format(target.type, visibility),
                       (x1-1,y1-1),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv.imwrite(os.path.join(out_dir.__str__(),image_filename), img)

if __name__ == "__main__":
    main()