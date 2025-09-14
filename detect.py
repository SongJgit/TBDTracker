import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml

from loguru import logger
import argparse

from tracker.utils.torch_utils import select_device
from tracker.utils.tools import save_results
from tracker.utils.visualization import plot_img, save_video
from tracker.utils.my_timer import Timer
from tracker.data.dataset import TestDataset

from tracker.detectors.model import DetectModel, default_imgsz


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='', help='./cfg/data_cfg/mot17-half-val.yaml')
    parser.add_argument('--detector', type=str, default='yolo', help='yolov7, yolox, etc.')
    """model path"""
    parser.add_argument('--weights', type=str, default='./weights/best.pt', help='model path')

    parser.add_argument('--img_size', type=int, default=None, help='image size, [h, w]')

    parser.add_argument('--conf_thresh', type=float, default=0.1, help='filter detections')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')

    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # other model path
    parser.add_argument('--save_dir', type=str, help='detect_results/{dataset_name}/{split}')
    parser.add_argument('--save_images', action='store_true', help='save detecting results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save detecting results (video)')

    # for yolox
    parser.add_argument('--yolox_exp_file', type=str, default='./tracker/yolox_utils/yolox_m.py')

    # for yolov7
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')

    return parser.parse_args()


def main(args, dataset_cfgs):
    """1. set some params"""

    # NOTE: if save video, you must save image
    if args.save_videos:
        args.save_images = True

    if args.img_size is None:
        args.img_size = default_imgsz(args.detector)

    save_dir = args.save_dir
    """2. load detector"""
    device = select_device(args.device)

    # TODO: support trt
    detector = DetectModel(
        detector=args.detector,
        model_weight=args.weights,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        img_size=args.img_size,
        device=device,
        yolox_exp_file=args.yolox_exp_file,
        classes=dataset_cfgs['CATEGORY_NAMES'],
    )
    """3. load sequences"""
    DATA_ROOT = dataset_cfgs['DATASET_ROOT']
    SPLIT = dataset_cfgs['SPLIT']

    seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'images', SPLIT)))
    seqs = [seq for seq in seqs if seq not in dataset_cfgs['IGNORE_SEQS']]
    if None not in dataset_cfgs['CERTAIN_SEQS']:
        seqs = dataset_cfgs['CERTAIN_SEQS']

    logger.info(f'Total {len(seqs)} seqs will be tracked: {seqs}')
    """4. Detecting"""
    timer = Timer()
    seq_fps = []

    for seq in seqs:
        logger.info(f'--------------Detecting seq {seq}--------------')

        dataset = TestDataset(DATA_ROOT,
                              SPLIT,
                              seq_name=seq,
                              img_size=detector.model_img_size,
                              model=args.detector,
                              stride=detector.stride)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        process_bar = enumerate(data_loader)
        process_bar = tqdm(process_bar, total=len(data_loader), ncols=150, dynamic_ncols=True)

        results = []

        for frame_idx, (ori_img, img, info) in process_bar:
            # start timing this frame
            timer.tic()

            img = detector.preprocess_img(img)
            ori_img = ori_img.squeeze(0)

            # get detector output
            output = detector.inference_with_postprocess(img, ori_img)

            if isinstance(output, torch.Tensor):
                output = output.detach().cpu().numpy()
            output[:, 2] -= output[:, 0]
            output[:, 3] -= output[:, 1]

            # save results
            scores = output[:, 4]
            tlwh = output[:, :4]
            categories = output[:, -1]
            results.append((frame_idx + 1, [-1] * len(output), tlwh, categories, scores))

            timer.toc()

            if args.save_images:
                plot_img(img=ori_img,
                         frame_id=frame_idx,
                         results=[tlwh, [-1] * len(output), categories],
                         save_dir=os.path.join(save_dir, SPLIT, seq, 'img1'))

        save_results(folder_name=os.path.join(save_dir, SPLIT, seq, 'det'), seq_name='det', results=results)

        # show the fps
        seq_fps.append(frame_idx / timer.total_time)
        logger.info(f'fps of seq {seq}: {seq_fps[-1]}')
        timer.clear()

        if args.save_videos:
            save_video(images_path=os.path.join(save_dir, SPLIT, 'vis_results', 'videos'))
            logger.info(f'save video of {seq} done')

        # show the average fps
    logger.info(f'average fps: {np.mean(seq_fps)}')


if __name__ == '__main__':

    args = get_args()

    with open(args.dataset, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    main(args, cfgs)
