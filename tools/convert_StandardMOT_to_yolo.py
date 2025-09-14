"""将UAVDT转换为yolo v5格式 class_id, xc_norm, yc_norm, w_norm, h_norm.

support mot17 and mot20, SNMOT
"""

import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np
import random
from half_dataset import cleanup_mot17
from enum import Enum
from typing import List
from tqdm import tqdm


class MOTClassesID(Enum):
    MOT = [
        dict(id=1, name='pedestrian'),  # only pedestrian
        dict(id=2, name='person_on_vehicle'),
        dict(id=3, name='car'),
        dict(id=4, name='bicycle'),
        dict(id=5, name='motorbike'),
        dict(id=6, name='non_mot_vehicle'),
        dict(id=7, name='static_person'),
        dict(id=8, name='distractor'),
        dict(id=9, name='occluder'),
        dict(id=10, name='occluder_on_ground'),
        dict(id=11, name='occluder_full'),
        dict(id=12, name='reflection'),
        dict(id=13, name='crowd'), ]
    SNMOT = [
        dict(id=-1, name='soccer_baller'), ]
    DanceTrack = [
        dict(id=1, name='dancer'), ]
    AntiUAVTrack = [dict(id=0, name='drone')]

    @classmethod
    def classes2id(cls, name):
        """_summary_

        Args:
            name (_type_): _description_

        Returns:
            _type_: {d'pedestrian': 1, 'person_on_vehicle': 2, ...}
        """
        CLASSES = cls.get_classes(name)
        return {c['name']: c['id'] for c in CLASSES}

    @classmethod
    def id2classes(cls, name, classes: List[str] | None = None):
        """_summary_

        Args:
            name (_type_): _description_
            classes (List[str] | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_:  {1: 'pedestrian', 2: 'person_on_vehicle', ...}
        """
        CLASSES2ID = cls.classes2id(name)
        if classes is None:
            classes = CLASSES2ID
        elif len(classes) == 0:
            classes = list(CLASSES2ID.keys())  # return all classes if classes is empty or None
        elif not isinstance(classes, list):
            classes = [classes]
        try:
            ID2CLASSES = {CLASSES2ID[cls]: cls for cls in classes}  # {1: 'pedestrian', 2: 'person_on_vehicle', ...}
        except Exception as e:
            raise ValueError(f'classes not in the dataset, available classes are {list(CLASSES2ID.keys())}, {e}')
        return ID2CLASSES

    @classmethod
    def get_classes(cls, dataset_name):
        if hasattr(cls, dataset_name):
            CLASSES = cls[dataset_name].value
        else:
            raise ValueError(f'Unsupported datasets_name: {dataset_name},',
                             f'Must be one of {list(cls.__members__.keys())}')
        return CLASSES


image_wh_dict = {}  # seq->(w,h) 字典 用于归一化


def generate_imgs_and_labels(opts):
    """产生图片路径的txt文件以及yolo格式真值文件."""
    if opts.split == 'test':
        seq_list = os.listdir(osp.join(opts.data_root, 'test'))
    else:
        if 'MOT17' in opts.data_root:
            cleanup_mot17(osp.join(opts.data_root, opts.split), keep_detection='FRCNN')
        seq_list = os.listdir(osp.join(opts.data_root, opts.split))

        # seq_list = [item for item in seq_list if 'FRCNN' in item]  # 只取一个FRCNN即可

    print('--------------------------')
    print(f'Total {len(seq_list)} seqs!!')
    print(seq_list)

    if opts.random:
        random.shuffle(seq_list)

    # 定义类别 MOT只有一类
    # CATEGOTY_ID = 0  # pedestrian

    # 定义帧数范围
    # frame_range = {'start': 0.0, 'end': 1.0}

    if opts.split == 'test':
        process_train_test(opts, seqs=seq_list, split=opts.split)
    else:
        process_train_test(opts, seqs=seq_list, split=opts.split)


def force_symlink(target: str, link_name: str) -> None:
    # 检查链接是否已存在
    if os.path.lexists(link_name):
        # 如果是链接，则删除它
        if os.path.islink(link_name):
            os.unlink(link_name)
        else:
            # 如果不是链接，则抛出异常或采取其他措施
            raise Exception(f'{link_name} 已存在且不是一个符号链接')
    # 创建新的符号链接
    os.symlink(target, link_name)


def generate_img_index_file(dst_folder, video_name, img_name, subset):
    to_file = os.path.join(f'./datasets/{osp.basename(opts.data_root)}/', subset + '.txt')
    with open(to_file, 'a') as f:

        f.write(osp.join(opts.data_root, 'images', subset, video_name, img_name) + '\n')

        f.close()


def process_train_test(opt: argparse.ArgumentParser, seqs: list, cat_id: int = 0, split: str = 'trian') -> None:
    """处理MOT17的train 或 test 由于操作相似 故另写函数."""

    for seq in tqdm(seqs, desc=f'Processing {split} sequences'):

        img_dir = osp.join(opt.data_root, split, seq, 'img1')  # 图片路径
        imgs = sorted(os.listdir(img_dir))  # 所有图片的相对路径
        # seq_length = len(imgs)  # 序列长度

        # 求解图片高宽
        img_eg = cv2.imread(osp.join(img_dir, imgs[0]))
        w0, h0 = img_eg.shape[1], img_eg.shape[0]  # 原始高宽

        ann_of_seq_path = os.path.join(img_dir, '../', 'gt', 'gt.txt')  # GT文件路径

        gt_to_path = osp.join(opt.data_root, 'labels', split, seq)  # 要写入的真值文件夹
        # 如果不存在就创建
        if not osp.exists(gt_to_path):
            os.makedirs(gt_to_path)

        exist_gts = []  # 初始化该列表 每个元素对应该seq的frame中有无真值框
        # 如果没有 就在train.txt产生图片路径

        for idx, img in enumerate(imgs):
            # img 形如: img000001.jpg

            # 第一步 产生图片软链接
            # print('step1, creating imgs symlink...')
            if opts.generate_imgs:
                img_to_path = osp.join(opt.data_root, 'images', split, seq)  # 该序列图片存储位置

                if not osp.exists(img_to_path):
                    os.makedirs(img_to_path)

                force_symlink(osp.abspath(osp.join(img_dir, img)), osp.join(img_to_path, img))

                # 第二步 产生真值文件, 有真值文件则读取并写入
            if osp.exists(os.path.join(img_dir, '../', 'gt', 'gt.txt')):
                # print('step2, generating gt files...')
                ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',')  # GT内容
                ann_of_current_frame = ann_of_seq[ann_of_seq[:, 0] == float(idx + 1), :]  # 筛选真值文件里本帧的目标信息
                exist_gts.append(True if ann_of_current_frame.shape[0] != 0 else False)

                gt_to_file = osp.join(gt_to_path, img[:-4] + '.txt')

                with open(gt_to_file, 'w') as f_gt:
                    for i in range(ann_of_current_frame.shape[0]):
                        # 筛选出置信度为1的, 0表示ignore
                        # 筛选出类别为1，或类别为-1(SNMOT)
                        # 筛选出可见度大于0.25, 或可见度为-1(SNMOT)
                        conf = int(ann_of_current_frame[i][6])
                        category_id = int(ann_of_current_frame[i][7])
                        visibility = float(ann_of_current_frame[i][8])
                        if conf == 1  \
                            and (category_id== 1 or category_id == -1 or category_id==0) \
                            and (visibility > 0.25 or visibility ==-1) :

                            # bbox xywh
                            x0, y0 = int(ann_of_current_frame[i][2]), int(ann_of_current_frame[i][3])
                            x0, y0 = max(x0, 0), max(y0, 0)
                            w, h = int(ann_of_current_frame[i][4]), int(ann_of_current_frame[i][5])

                            xc, yc = x0 + w // 2, y0 + h // 2  # 中心点 x y

                            # 归一化
                            xc, yc = xc / w0, yc / h0
                            xc, yc = min(xc, 1.0), min(yc, 1.0)
                            w, h = w / w0, h / h0
                            w, h = min(w, 1.0), min(h, 1.0)
                            assert w <= 1 and h <= 1, f'{w}, {h} must be normed, original size{w0}, {h0}'
                            assert xc >= 0 and yc >= 0, f'{x0}, {y0} must be positve'
                            assert xc <= 1 and yc <= 1, f'{x0}, {y0} must be le than 1'
                            category_id = cat_id

                            write_line = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(category_id, xc, yc, w, h)

                            f_gt.write(write_line)

                f_gt.close()
            else:
                Warning(f'{split} {seq} has no gt file, will skip it')

        # 第三步 产生图片索引train.txt等
        # print(f'generating img index file of {seq}')
        to_file = os.path.join(opt.folder_to_txt, osp.basename(opt.data_root), split + '.txt')
        with open(to_file, 'a+') as f:
            for idx, img in enumerate(imgs):

                if split == 'test' or exist_gts[idx]:
                    # f.write(osp.abspath(osp.join(opt.data_root, 'images', split, seq, img + '\n')))
                    f.write(osp.join(opt.data_root, 'images', split, seq, img + '\n'))
            f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/MOT_Datasets/MOT20')
    # parser.add_argument('--datasets_name',
    #                     type=str,
    #                     required=True,
    #                     default='MOT',
    #                     help='support MOT17/20, DanceTrack, VisDroneTrack, and SNMOT(SoccerNet MOT)')
    # parser.add_argument('--classes',
    #                     nargs='+',
    #                     default=['pedestrian'],
    #                     help='input multiple items')
    parser.add_argument('--split', type=str, default='train', help='train, test or val')
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--folder_to_txt', type=str, default='datasets', help='folder to save the images index txt')
    parser.add_argument('--certain_seqs', action='store_true', help='for debug')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of test dataset divide train dataset')
    parser.add_argument('--random', action='store_true', help='random split train and test')

    opts = parser.parse_args()
    if not osp.exists(f'./{opts.folder_to_txt}/{osp.basename(opts.data_root)}'):
        os.makedirs(f'./{opts.folder_to_txt}/{osp.basename(opts.data_root)}')
    generate_imgs_and_labels(opts)
    # python tools/convert_MOT17_to_yolo.py --data_root /data/MOT_Datasets/MOT17-50 --split train --generate_imgs
