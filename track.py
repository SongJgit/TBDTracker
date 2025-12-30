import os
import numpy as np
import torch
from mmengine import Config, fileio
from mmengine import MODELS, DictAction
from tqdm import tqdm
import yaml
from tracker.utils.logger_config import global_logger as logger
import argparse

from tracker.utils import TRACKED_RESULTS_ROOT, DATA_CFG_ROOT, EVAL_CFG_ROOT, TRACKER_CFG_ROOT
from tracker.utils.torch_utils import select_device
from tracker.utils.tools import save_results
from tracker.utils.visualization import plot_img, save_video, plot_img_vis
from tracker.utils.my_timer import Timer
from tracker.data.dataset import TestDataset
from tracker.detectors.model import DetectModel
import os.path as osp


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='mot17')

    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')
    parser.add_argument('--cfg_options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    parser.add_argument('--motion_model_path', type=str, default=None, help='path for motion model path')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')
    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')
    parser.add_argument('--save_folder', type=str, default='track_results/{tracker_name}/{dataset_name}/{split}')

    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')
    parser.add_argument('--classes', nargs='+', type=int, default=[0],
                        help='filter by class: --classes 0')
    parser.add_argument('--eval_yaml', type=str, default=None, help='eval yaml')
    """camera parameter"""
    parser.add_argument('--camera_parameter_folder',
                        type=str,
                        default='./tracker/cam_param_files',
                        help='folder path of camera parameter files')

    subparsers = parser.add_subparsers(dest='mode',
                                       required=True,
                                       help='Select the detections source, file or detector.')

    detected_parser = subparsers.add_parser(
        'file', help='detections from detector or folder(standard MOT files format like MOT17)')
    detected_parser.add_argument('--detected_folder',
                                 type=str,
                                 default=None,
                                 help='./MOT17/train, test, val. get det.txt')

    detector_parser = subparsers.add_parser('detector', help='detections from detector')
    detector_parser.add_argument('--detector', type=str, default=None, help='yolov7, yolox, etc.')
    detector_parser.add_argument('--det_thresh', type=float, default=0.1, help='filter detections')
    detector_parser.add_argument('--nms_thresh', type=float, default=0.45, help='thresh for NMS')
    detector_parser.add_argument('--detector_model_path', type=str, default='./weights/best.pt', help='model path')
    detector_parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')
    """yolox"""
    detector_parser.add_argument('--yolox_exp_file', type=str, default='./tracker/yolox_utils/yolox_m.py')
    """tensorrt options"""
    detector_parser.add_argument('--trt', action='store_true', help='use tensorrt engine to detect and reid')
    detector_parser.add_argument('--img_size', type=int, default=1280, help='image size, [h, w]')


    return parser.parse_args()


def main(args, dataset_cfg):

    # NOTE: if save video, you must save image
    if args.save_videos:
        args.save_images = True
    """2. load detector or motion model"""
    device = select_device(args.device)

    if args.motion_model_path is not None and len(args.motion_model_path) != 0:
        # load learning-aided Kalman filter
        try: 
            from filternet.utils import attempt_load_model
        except:
            raise ImportError('filternet not installed.')
            
        motion_model = attempt_load_model(args.motion_model_path)
        motion_model.eval()
        motion_model_cfg = dict(type='knet', motion_model=motion_model)
    else:
        motion_model_cfg = None

    if args.mode == 'file':
        args.detector = None
        args.img_size = None
    else:
        # adjust tensorrt
        if args.detector_model_path.endswith('.engine'):
            args.trt = True
        detector = DetectModel(detector=args.detector,
                               model_weight=args.detector_model_path,
                               conf_thresh=args.conf_thresh,
                               nms_thresh=args.nms_thresh,
                               img_size=args.img_size,
                               device=device,
                               yolox_exp_file=args.yolox_exp_file,
                               classes=dataset_cfg['CATEGORY_NAMES'],
                               use_trt=args.use_trt)
    """3. load sequences"""

    DATA_ROOT = dataset_cfg['DATASET_ROOT']
    SPLIT = dataset_cfg['SPLIT']

    seqs = sorted(os.listdir(os.path.join(DATA_ROOT, 'images', SPLIT)))
    seqs = [seq for seq in seqs if seq not in dataset_cfg['IGNORE_SEQS']]
    if None not in dataset_cfg['CERTAIN_SEQS']:
        seqs = dataset_cfg['CERTAIN_SEQS']

    logger.info(f'Total {len(seqs)} seqs will be tracked: {seqs}')

    # save_dir = f'./results_tracked/{args.dataset}/{args.tracker}/{args.save_folder}'
    save_dir = osp.join(TRACKED_RESULTS_ROOT, args.dataset, args.tracker, args.save_folder)

    tracker_cfg = Config.fromfile(osp.join(TRACKER_CFG_ROOT, f'{args.tracker}.py'))
    if args.cfg_options is not None:
        tracker_cfg.merge_from_dict(args.cfg_options)
    logger.info(tracker_cfg.pretty_text)
    """4. Tracking"""
    # set timer
    timer = Timer()
    seq_fps = []

    for seq in seqs:
        # logger.info(f'--------------tracking seq {seq}--------------')

        if args.mode == 'file':
            dataset = TestDataset(DATA_ROOT,
                                  SPLIT,
                                  seq_name=seq,
                                  img_size=None,
                                  model=None,
                                  stride=None,
                                  det_folder=args.detected_folder)
        else:
            dataset = TestDataset(DATA_ROOT,
                                  SPLIT,
                                  seq_name=seq,
                                  img_size=detector.model_img_size,
                                  model=args.detector,
                                  stride=detector.stride)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # store the seq name, for conveniently reading the camera param file w.r.t. each sequence
        args.cam_param_file = os.path.join(args.camera_parameter_folder, args.dataset, seq + '.txt')

        # Preloading the motion_model can save time by avoiding repeated loading.
        if motion_model_cfg is not None:
            tracker_cfg.tracker.motion_format = motion_model_cfg
        tracker = MODELS.build(tracker_cfg.tracker)

        process_bar = enumerate(data_loader)
        process_bar = tqdm(process_bar, total=len(data_loader), ncols=150, dynamic_ncols=True)

        results = []

        for frame_idx, (ori_img, img, info) in process_bar:
            timer.tic()
            # start timing this frame
            ori_img = ori_img.squeeze(0)

            if args.mode == 'file':
                output = info['dets'].squeeze(0)
            else:
                # output: (tlbr, conf, cls)
                img = detector.preprocess_img(img)
                output = detector.inference_with_postprocess(img, ori_img)
                output[:, 2] -= output[:, 0]
                output[:, 3] -= output[:, 1]  # convert tlbr to tlwh

            if isinstance(output, torch.Tensor):
                output = output.detach().cpu().numpy()
                # save results

            output = output[np.isin(output[:, -1], args.classes)]

            current_tracks = tracker.update(output, img, ori_img.cpu().numpy())
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            for trk in current_tracks:
                bbox = trk.tlwh
                id = trk.track_id
                cls = trk.category
                score = trk.score

                # if cls !=0: continue
                # filter low area bbox
                if bbox[2] * bbox[3] > args.min_area:
                    cur_tlwh.append(bbox)
                    cur_id.append(id)
                    cur_cls.append(cls)
                    cur_score.append(score)
                    # results.append((frame_id + 1, id, bbox, cls))

            results.append((frame_idx + 1, cur_id, cur_tlwh, cur_cls, cur_score))

            timer.toc()

            if args.save_images:
                saved_images_path = os.path.join(save_dir, SPLIT, 'vis_results', 'images', seq)
                plot_img_vis(img=ori_img,
                             frame_id=frame_idx,
                             results=[cur_tlwh, cur_id, cur_cls],
                             save_dir=os.path.join(saved_images_path))
                # plot_img(img=ori_img,
                #          frame_id=frame_idx,
                #          results=[cur_tlwh, cur_id, cur_cls],
                #          save_dir=os.path.join(saved_images_path))

        save_results(folder_name=os.path.join(save_dir, SPLIT), seq_name=seq, results=results)

        # # show the fps
        seq_fps.append(frame_idx / timer.total_time)
        logger.info(f'fps of seq {seq}: {seq_fps[-1]}')
        timer.clear()

        if args.save_videos and args.save_images:
            video_path = saved_images_path.replace('/images/', '/videos/') + '.mp4'
            save_video(video_path, saved_images_path)
            logger.info(f'save video of {seq} done: {video_path}')

    # show the average fps
    logger.info(f'average fps: {np.mean(seq_fps)}')

    if args.eval_yaml is not None:
        from run_custom_dataset_eval import main as eval_main
        with open(osp.join(EVAL_CFG_ROOT, f'{args.eval_yaml}.yaml'), 'r') as f:
            yaml_dataset_config = yaml.safe_load(f)
        yaml_dataset_config['tracker_structure_config']['trackers_folder'] = save_dir
        yaml_dataset_config['tracker_structure_config']['split_name'] = SPLIT
        yaml_dataset_config['gt_structure_config']['train_or_test'] = SPLIT
        eval_main(None, yaml_dataset_config)
        logger.info('Evaluation done')
        logger.info(f'Tracked results saved to {save_dir}/{SPLIT}')
        logger.info(f"Eval track from {yaml_dataset_config['tracker_structure_config']['trackers_folder']}/{SPLIT}")

    if not osp.exists(os.path.join(save_dir, SPLIT, 'configs')):
        os.makedirs(os.path.join(save_dir, SPLIT, 'configs'))

    if motion_model_cfg is not None:
        tracker_cfg.tracker.motion_format.motion_model = args.motion_model_path
    tracker_cfg.dump(os.path.join(save_dir, SPLIT, 'configs', 'tracker_cfg.py'))
    fileio.dump(dataset_cfg, os.path.join(save_dir, SPLIT, 'configs', 'dataset_cfg.yaml'))
    if args.eval_yaml is not None:
        fileio.dump(yaml_dataset_config, os.path.join(save_dir, SPLIT, 'configs', 'eval.yaml'))
    logger.info(f'Save configs to {save_dir}/{SPLIT}/configs\n'
                f'Total time: {timer.total_time:.2f}s\n'
                f'All done')


if __name__ == '__main__':

    args = get_args()

    with open(osp.join(DATA_CFG_ROOT, f'{args.dataset}.yaml'), 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    main(args, cfgs)
