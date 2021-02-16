import cv2 as cv
import numpy as np
import yaml
from pathlib import Path
import sys
import os
import numpy
import glob
from mot_io import MotGt, MotDet

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (255, 255, 0)

font                   = cv.FONT_HERSHEY_COMPLEX
fontScale              = 0.4
fontColor              = COLOR_WHITE
lineType               = 1
shadowColor = COLOR_BLACK


def fit_coordinate_in_box(coord, lower_bound, upper_bound):
    coord = int(coord)
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

    parser.add_argument('--input-type', metavar='IT', type=str,
                        default=None,
                        help='Either \"gt\" or \"det\"')

    parser.add_argument('--input-file', metavar='GTF', type=str,
                        default=None,
                        help='Enter the full path of the MOT ground-truth or detection file')


    args = parser.parse_args()


    image_folder = Path(args.image_folder)
    out_dir = Path(args.output_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    input_file_path = Path(args.input_file)
    input_type = args.input_type
    assert input_type in ["gt","det"], "Error: Input type should either be \"gt\" or \"det\"!"

    mot_container = None
    if input_type == 'gt':
        mot_container = MotGt(input_file_path.__str__())
    else:
        mot_container = MotDet(input_file_path.__str__())

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

        targets_to_draw = mot_container.get_objects_in_frame(frame_id)

        for target in targets_to_draw:
            _, bbox, visibility_or_conf_score = target.get_state_in_frame(frame_id)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            x1 = fit_coordinate_in_box(x1, 0, frame_width)
            x2 = fit_coordinate_in_box(x2, 0, frame_width)
            y1 = fit_coordinate_in_box(y1, 0, frame_height)
            y2 = fit_coordinate_in_box(y2, 0, frame_height)

            if input_type == 'gt':
                box_color = COLOR_RED if target.activity == 0 else COLOR_GREEN
                label = "id:{}, t:{}, v:{:4.3f}".format(target.id, target.type, visibility_or_conf_score)
            else:
                box_color = COLOR_CYAN
                label = "c:{:4.3f}".format(visibility_or_conf_score)

            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

            # original label
            cv.putText(img, label,
                       (x1, y1),
                       font,
                       fontScale,
                       shadowColor,
                       lineType)
            # shadow (to improve readability)
            cv.putText(img, label,
                       (x1-1,y1-1),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv.imwrite(os.path.join(out_dir.__str__(),image_filename), img)

if __name__ == "__main__":
    main()