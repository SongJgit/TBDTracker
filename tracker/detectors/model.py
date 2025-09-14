from loguru import logger
import torch
from typing import List, Any, Union
import numpy as np

try:
    from ..accelerations.tensorrt_tools import TensorRTConverter, TensorRTInference
except Exception as e:
    logger.warning(e)
    logger.warning('Load TensorRT fail. If you want to convert model to TensorRT, please install the packages.')

UL_MODELS = ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolo12', 'rtdetr', 'sam']
YOLOX = ['yolox']

DETECT_MODELS = UL_MODELS + YOLOX


def is_ultralytics_model(yolo_name):
    return any(yolo in str(yolo_name) for yolo in UL_MODELS)


def is_yolox_model(yolo_name):
    return 'yolox' in str(yolo_name)


def default_imgsz(yolo_name):
    if is_ultralytics_model(yolo_name):
        return [640, 640]
    elif is_yolox_model(yolo_name):
        return [800, 1440]
    else:
        return [640, 640]


def get_yolo_inferer(yolo_model: str):

    if is_yolox_model(yolo_model):
        try:
            import yolox  # for linear_assignment
            from yolox.exp import get_exp
            from detectors.yolox_utils.postprocess import postprocess_yolox
            from yolox.utils import fuse_model
            assert yolox.__version__
        except (ImportError, AssertionError, AttributeError) as e:
            logger.warning(e)
            logger.warning('Load yolox fail. If you want to use yolox, please check the installation.')

    elif is_ultralytics_model(yolo_model):
        try:
            from ultralytics import YOLO
            from ul_yolo_utils.postprocess import postprocess as postprocess_ul_yolo
        # ultralytics already installed when running track.py
        except (ImportError, AssertionError, AttributeError) as e:
            logger.warning(e)
            logger.warning(
                'Load ultralytics yolo fail. If you want to use ultralytics yolo, please check the installation.')
    else:
        logger.error('Failed to infer inference mode from yolo model name')
        logger.error('Your model name has to contain either yolox, yolo_nas or yolov8')
        exit()


class DetectModel:

    def __init__(
        self,
        detector: str,
        model_weight: str,
        conf_thresh: float,
        nms_thresh: float,
        img_size: list[int, int],
        classes=None,
        device: str = '0',
        stride: int = None,
        yolox_exp_file: str = None,
        use_trt: bool = False,
    ):
        # get_yolo_inferer(detector)
        self.detector = detector
        self.classes = classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        self.device = device
        self.use_trt = use_trt
        if is_yolox_model(detector):
            try:
                import yolox  # for linear_assignment
                from yolox.exp import get_exp
                from .yolox_utils.postprocess import postprocess_yolox
                from yolox.utils import fuse_model
                assert yolox.__version__
            except (ImportError, AssertionError, AttributeError) as e:
                logger.warning(e)
                logger.warning('Load yolox fail. If you want to use yolox, please check the installation.')

            self.postporcess = postprocess_yolox
            exp = get_exp(yolox_exp_file, None)  # TODO: modify num_classes etc. for specific dataset
            model_img_size = exp.input_size
            model = exp.get_model()
            model.to(self.device)
            model.eval()

            if self.use_trt:  # convert trt
                # check if need to convert
                if not model_weight.endswith('.engine'):
                    trt_converter = TensorRTConverter(model,
                                                      input_shape=[3, *model_img_size],
                                                      ckpt_path=model_weight,
                                                      min_opt_max_batch=[1, 1, 1],
                                                      device=device,
                                                      load_ckpt=True,
                                                      ckpt_key='model')
                    trt_converter.export()
                    model = TensorRTInference(engine_path=trt_converter.trt_model,
                                              min_opt_max_batch=[1, 1, 1],
                                              device=device)
                else:
                    model = TensorRTInference(engine_path=model_weight, min_opt_max_batch=[1, 1, 1], device=device)

            else:  # normal load

                logger.info(f'loading detector {detector} checkpoint {model_weight}')
                ckpt = torch.load(model_weight, map_location=device)
                model.load_state_dict(ckpt['model'])
                logger.info('loaded checkpoint done')
                model = fuse_model(model)

            stride = None  # match with yolo v7

            logger.info(f'Now detector is on device {next(model.parameters()).device}')

        elif is_ultralytics_model(detector):
            try:
                from ultralytics import YOLO
                from .ul_yolo_utils.postprocess import postprocess as postprocess_ul_yolo
            # ultralytics already installed when running track.py
            except (ImportError, AssertionError, AttributeError) as e:
                logger.warning(e)
                logger.warning(
                    'Load ultralytics yolo fail. If you want to use ultralytics yolo, please check the installation.')

            self.postporcess = postprocess_ul_yolo
            model_img_size = self.img_size
            if self.use_trt:
                # for ultralytics, we use the api provided by official ultralytics
                # check if need to convert
                if not model_weight.endswith('.engine'):
                    model = YOLO(model_weight)
                    model = YOLO(model.export(format='engine'))
                else:
                    model = YOLO(model_weight)

            else:
                logger.info(f'loading detector {detector} checkpoint {model_weight}')
                model = YOLO(model_weight)

                logger.info('loaded checkpoint done')

        else:
            logger.error(f'detector {detector} is not supprted')
            logger.error(f'If you want to use the yolo v8 by ultralytics, please specify the `--detector` \
                        as the string including the substring , \
                        such as {DETECT_MODELS}')
            exit(0)

        stride = None

        self.stride = stride
        self.model_img_size = model_img_size
        self.model = model.to(device)

    def inference(self, img: torch.tensor) -> Any:
        with torch.no_grad():
            if is_ultralytics_model(self.detector):
                return self.model.predict(img, conf=self.conf_thresh, iou=self.nms_thresh, imgsz=self.img_size)
            else:
                return self.model(img)

    def inference_with_postprocess(self, img: torch.tensor, ori_img) -> Any:
        pred = self.inference(img)
        if is_ultralytics_model(self.detector):
            return self.postporcess(pred)
        elif is_yolox_model(self.detector):
            return self.postporcess(pred,
                                    len(self.classes),
                                    self.conf_thresh,
                                    self.nms_thresh,
                                    img=img,
                                    ori_img=ori_img)
        else:
            raise NotImplementedError

    def preprocess_img(self, img: torch.tensor) -> Union[torch.tensor, np.ndarray]:
        if is_ultralytics_model(self.detector):
            return img.squeeze(0).cpu().numpy()
        else:
            return img.to(self.device).float()  # (1, C, H, W)
