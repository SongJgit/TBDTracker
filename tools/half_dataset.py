from pathlib import Path
import pandas as pd
import argparse
import os
import shutil

# Modified from https://github.com/mikel-brostrom/boxmot/blob/master/tracking/val.py


def split_dataset(src_fldr: Path, mode='link', percent_to_split: float = 0.5) -> None:
    """Copies the dataset to a new location and removes a specified percentage
    of images and annotations, adjusting the frame index to start at 1.

    Args:
        src_fldr (Path): Source folder containing the dataset.
        percent_to_delete (float): Percentage of images and annotations to remove.
    """
    # Ensure source path is a Path object
    src_fldr = Path(src_fldr)

    # Generate the destination path by replacing "MOT17" with "MOT17-half" in the source path
    new_benchmark_name = f'{opt.benchmark}-tr{int(percent_to_split * 100)}-val{int((1-percent_to_split) * 100)}'
    train_dst_fldr = Path(str(src_fldr).replace(f'{opt.benchmark}', new_benchmark_name))
    val_dst_fldr = Path(str(train_dst_fldr).replace('train', 'val'))

    # Copy the dataset to a new location manually using pathlib if it doesn't already exist
    if not train_dst_fldr.exists():
        train_dst_fldr.mkdir(parents=True)
        val_dst_fldr.mkdir(parents=True)
        for item in src_fldr.rglob('*'):
            if item.is_dir():
                (train_dst_fldr / item.relative_to(src_fldr)).mkdir(parents=True, exist_ok=True)
                (val_dst_fldr / item.relative_to(src_fldr)).mkdir(parents=True, exist_ok=True)
            elif item.is_file() and item.suffix == '.ini':
                shutil.copy2(item, train_dst_fldr / item.relative_to(src_fldr))
                shutil.copy2(item, val_dst_fldr / item.relative_to(src_fldr))

    # List all sequences in the destination folder
    src_seq_paths = [f for f in src_fldr.iterdir() if f.is_dir()]
    train_dst_seq_paths = [f for f in train_dst_fldr.iterdir() if f.is_dir()]
    val_dst_seq_paths = [f for f in val_dst_fldr.iterdir() if f.is_dir()]

    for src_seq_path, train_dst_seq_path, val_dst_seq_path in zip(src_seq_paths, train_dst_seq_paths,
                                                                  val_dst_seq_paths):
        src_seq_gt_path = src_seq_path / 'gt' / 'gt.txt'
        src_seq_det_path = src_seq_path / 'det' / 'det.txt'
        src_jpg_folder_path = src_seq_path / 'img1'

        dst_train_seq_gt_path = train_dst_seq_path / 'gt' / 'gt.txt'
        dst_train_seq_det_path = train_dst_seq_path / 'det' / 'det.txt'
        dst_train_jpg_folder_path = train_dst_seq_path / 'img1'
        dst_val_seq_gt_path = val_dst_seq_path / 'gt' / 'gt.txt'
        dst_val_seq_det_path = val_dst_seq_path / 'det' / 'det.txt'
        dst_val_jpg_folder_path = val_dst_seq_path / 'img1'

        if not src_seq_gt_path.exists():
            print(f'Ground truth file not found for {dst_train_seq_gt_path}. Skipping...')
            continue

        df = pd.read_csv(src_seq_gt_path, sep=',', header=None)
        nr_seq_imgs = df[0].unique().max()
        split = int(nr_seq_imgs * percent_to_split)

        # Check if the sequence is already split
        if nr_seq_imgs <= split:
            print(f'Sequence {src_seq_path} already split. Skipping...')
            continue

        print(f'Number of annotated frames in {src_seq_path}: frame 1 to {split} for training and',
              f'frame {split + 1} to {nr_seq_imgs} for val')

        # Split the
        train_df = df[df[0] <= split].copy()
        train_df.to_csv(dst_train_seq_gt_path, header=None, index=None, sep=',')
        val_df = df[df[0] > split].copy()
        val_df[0] = df[0] - split
        val_df.to_csv(dst_val_seq_gt_path, header=None, index=None, sep=',')

        if src_seq_det_path.exists():
            det_df = pd.read_csv(src_seq_det_path, sep=',', header=None)
            det_df = det_df.sort_values(by=0)
            train_det_df = det_df[det_df[0] <= split].copy()
            train_det_df.to_csv(dst_train_seq_det_path, header=None, index=None, sep=',')
            val_det_df = det_df[det_df[0] > split].copy()
            val_det_df[0] = det_df[0] - split
            val_det_df.to_csv(dst_val_seq_det_path, header=None, index=None, sep=',')
        else:
            print(f'Detection file not found for {dst_train_seq_det_path}. Skipping...')

        src_jpg_paths = list(src_jpg_folder_path.glob('*.jpg'))
        for src_jpg_path in src_jpg_paths:
            frame_number = int(src_jpg_path.stem)
            if frame_number <= split:
                new_index = frame_number
                new_jpg_name = f'{new_index:06}.jpg'
                dst_jpg_path = dst_train_jpg_folder_path / new_jpg_name
            else:
                new_index = frame_number - split
                new_jpg_name = f'{new_index:06}.jpg'
                dst_jpg_path = dst_val_jpg_folder_path / new_jpg_name

            if mode == 'copy':
                shutil.copy2(src_jpg_path, dst_jpg_path)
            else:
                if dst_jpg_path.is_symlink():
                    dst_jpg_path.unlink()
                dst_jpg_path.symlink_to(src_jpg_path)

    print(f'Done splitting, save to {train_dst_fldr.parent}')


def cleanup_mot17(data_dir, keep_detection='FRCNN'):
    """Cleans up the MOT17 dataset to resemble the MOT16 format by keeping only
    one detection folder per sequence. Skips sequences that have already been
    cleaned.

    Args:
    - data_dir (str): Path to the MOT17 train directory.
    - keep_detection (str): Detection type to keep (options: 'DPM', 'FRCNN', 'SDP'). Default is 'DPM'.
    """

    # Get all folders in the train directory
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Identify unique sequences by removing detection suffixes
    unique_sequences = set(seq.split('-')[0] + '-' + seq.split('-')[1] for seq in all_dirs)

    for seq in unique_sequences:
        # Directory path to the cleaned sequence
        cleaned_seq_dir = os.path.join(data_dir, seq)

        # Skip if the sequence is already cleaned
        if os.path.exists(cleaned_seq_dir):
            print(f'Sequence {seq} is already cleaned. Skipping.')
            continue

        # Directories for each detection method
        seq_dirs = [os.path.join(data_dir, d) for d in all_dirs if d.startswith(seq)]

        # Directory path for the detection folder to keep
        keep_dir = os.path.join(data_dir, f'{seq}-{keep_detection}')

        if os.path.exists(keep_dir):
            # Move the directory to a new name (removing the detection suffix)
            shutil.move(keep_dir, cleaned_seq_dir)
            print(f'Moved {keep_dir} to {cleaned_seq_dir}')

            # Remove other detection directories
            for seq_dir in seq_dirs:
                if os.path.exists(seq_dir) and seq_dir != keep_dir:
                    shutil.rmtree(seq_dir)
                    print(f'Removed {seq_dir}')
        else:
            print(f'Directory for {seq} with {keep_detection} detection does not exist. Skipping.')

    print('MOT17 Cleanup completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global arguments
    parser.add_argument('--source', type=Path, default='/data/MOT_Datasets/MOT17/train', help='Datasets path')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select the copy or link to src')
    link_parser = subparsers.add_parser('link', help='link to src')
    copy_parser = subparsers.add_parser('copy', help='copy from src')

    opt = parser.parse_args()
    source_path = Path(opt.source)
    opt.benchmark, opt.split = source_path.parent.name, source_path.name
    if opt.benchmark == 'MOT17':
        cleanup_mot17(opt.source)
    split_dataset(opt.source, mode=opt.mode, percent_to_split=0.5)
