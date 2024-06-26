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
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import Predictor


# constants
WINDOW_NAME = "GRiT"
MODEL_VERSION = 1
TRANSLATOR = 1
KEY = 'AAAAC3NzaC1lZDI1NTE5AAAAIPRITl0dLRphht4xYaOjfWOnq99mK1TYMuQGwFkQmYic'

def create_app():
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
            default="configs/GRiT_B_DenseCap_ObjectDet.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
        parser.add_argument(
            "--output",
            help="A file or directory to save output visualizations. "
            "If not given, will not save output",
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
            default='DenseCap',
            help="Choose a task to have GRiT perform",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=['MODEL.WEIGHTS' , 'models/grit_b_densecap_objectdet.pth'],
            nargs=argparse.REMAINDER,
        )
        return parser

    @app.route('/predict')
    def predictGet():
        return 'ok predict!'
    
    @app.route('/')
    def indexGet():
        return 'ok index!'

    @app.route('/predict', methods=['POST'])
    def predict():
        request_key = request.form.get('key')
        if (request_key != KEY):
            return 'wrong key', 401

        filename = request.files['image'].filename
        file_stream = request.files['image'].stream
        image = Image.open(request.files['image'].stream)
        if (image != None):
            print('hey')
            print(filename)

        image = _apply_exif_orientation(image)
        img = convert_PIL_to_numpy(image, format="BGR")

        #img = read_image(file_stream, format="BGR")



        # RGB > BGR
        # Можно не париться и задать input_format RGB, оно сделает это в модели


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

        if (args.output != None):
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
        descriptions = instances.pred_object_descriptions.data
        concatedDescriptions = '; '.join(descriptions)
        descriptionsRu = translator.translate(concatedDescriptions).split("; ")

        result['descriptionsEn'] = descriptions
        result['descriptionsRu'] = descriptionsRu
        result['modelVersion'] = MODEL_VERSION
        result['translator'] = TRANSLATOR


        return result
    
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.setLevel(logging.INFO)
    # logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    pred = Predictor(cfg)
    cpu_device = torch.device("cpu")
    translator = Translator(from_lang="en", to_lang='ru', email="nemilov220@gmail.com")
    return app
