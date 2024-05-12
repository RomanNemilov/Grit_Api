import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import json

import logging
import torch
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image
from translate import Translator

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import Predictor


# constants
WINDOW_NAME = "GRiT"
MODEL_VERSION = 1
TRANSLATOR = 1
app = Flask(__name__)

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

@app.route('/predict')
def predictGet():
    return 'ok!'

@app.route('/predict', methods=['POST'])
def predict():
    filename = request.files['image'].filename
    image = Image.open(request.files['image'].stream)
    if (image != None):
        print('hey')
        print(filename)

    image = image.convert('RGB')
    img = np.asarray(image)

    # если будет баг с ориентацией изображения решение здесь detectron2>data>detection_utils строка 119

    # RGB > BGR
    # Можно не париться и задать input_format RGB, оно сделает это в модели
    img = img[:, :, ::-1]

    start_time = time.time()
    predictions, visualized_output = pred.run_on_image(img)
    # logger.info(
    #     "{}: {} in {:.2f}s".format(
    #         filename,
    #         "detected {} instances".format(len(predictions["instances"]))
    #         if "instances" in predictions
    #         else "finished",
    #         time.time() - start_time,
    #     )
    # )

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if os.path.isdir(args.output):
        assert os.path.isdir(args.output), args.output
        out_path = os.path.join(args.output, filename)
    else:
        assert len(args.input) == 1, "Please specify a directory with args.output"
        out_path = args.output
    visualized_output.save(out_path)
    print('INSTANCE DESCRIPRIONS START HERE')
    instances = predictions["instances"].to(cpu_device)
    print(instances)

    result = {}
    description = instances.pred_object_descriptions.data
    descriptionRu = []
    for prediction in description:
        descriptionRu.append(translator.translate(prediction))
    result['descriptionsEn'] = description
    result['descriptionsRu'] = descriptionRu
    result['modelVersion'] = MODEL_VERSION
    result['translator'] = TRANSLATOR


    return result

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.setLevel(logging.INFO)
    # logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    pred = Predictor(cfg)
    cpu_device = torch.device("cpu")
    translator = Translator(from_lang="en", to_lang='ru')

    app.run(debug=True, port=8888, host="0.0.0.0")