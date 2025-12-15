"""run 2D MOT custom dataset with config file."""

import sys
import os
import argparse
from multiprocessing import freeze_support
from typing import Union
import os.path as osp
import numpy as np
import yaml
from tracker.evaluator.custom_dataset import CustomDataset
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402
from tabulate import tabulate


def has_None(seq_info: Union[dict, None]) -> bool:
    if seq_info is None:
        return True
    for k, v in seq_info.items():
        if v is None:
            return True


def extract_combined_metrics(output_res):
    """_summary_

    Args:
        output_res (_type_):
        output_res = {
        'CustomDataset': {
            '/data/Project/.../val': {
                'COMBINED_SEQ': {
                    'pedestrian': {
                        'HOTA': {'HOTA(0)': 0.78725578, 'AssA': [0.75354833, ...]},
                        'CLEAR': {'MOTA': 0.77978239, 'IDSW': 215, ...},
                        'Identity': {'IDF1': 0.77995643, ...}
                    }
                }
            }
        }
    }

    Returns:
        _type_: _description_
    """
    # Metrics
    metrics = {}
    dataset_name = [name for name in output_res.keys()]
    for name in dataset_name:
        combined_data = output_res[name]
        metrics[name] = {}
        result_path = [path for path in combined_data.keys()]

        for path in result_path:
            labels = [label for label in combined_data[path]['COMBINED_SEQ'].keys()]
            metrics[name][path] = {}
            for label in labels:
                combined = combined_data[path]['COMBINED_SEQ'][label]
                metrics[name][path][label] = {
                    'HOTA': 100 * combined['HOTA']['HOTA'].mean(),
                    'DetA': 100 * combined['HOTA']['DetA'].mean(),
                    'AssA': 100 * combined['HOTA']['AssA'].mean(),
                    'MOTA': 100 * combined['CLEAR']['MOTA'],
                    'IDF1': 100 * combined['Identity']['IDF1'],
                    'IDSW': combined['CLEAR']['IDSW'], }
    return metrics


def main(args, yaml_dataset_config):
    freeze_support()

    # with open(args.config_path, 'r') as f:
    #     yaml_dataset_config = yaml.safe_load(f)

    # Adaptive sequence length to SEQ_INFO
    hasNone = has_None(yaml_dataset_config['SEQ_INFO'])
    gt_structure_config = yaml_dataset_config['gt_structure_config']
    if hasNone:
        if gt_structure_config['has_split']:
            if gt_structure_config['train_or_test'] == 'train':
                train_split_fol = gt_structure_config['train_folder_name']
            else:
                train_split_fol = gt_structure_config['test_folder_name']
        else:
            train_split_fol = ''
        split_name = train_split_fol
        seqs = os.listdir(osp.join(gt_structure_config['data_root'], train_split_fol))
        if yaml_dataset_config['SEQ_INFO'] is not None:
            for k, v in yaml_dataset_config['SEQ_INFO'].items():
                if k in seqs and v is None:
                    gt_txt_name = gt_structure_config['gt_txt_name']
                    if 'seq_name' in gt_txt_name:
                        gt_txt_name = gt_txt_name.format(seq_name=k)
                    gt_txt = gt_structure_config['gt_loc_format'].format(
                        data_root=gt_structure_config['data_root'],
                        split_name=split_name,
                        seq_name=k,
                        gt_folder_name=gt_structure_config['gt_folder_name'],
                        gt_txt_name=gt_txt_name)
                    gt_tracks = np.loadtxt(gt_txt, delimiter=',')
                    max_frame = gt_tracks[:, 0].max()
                    yaml_dataset_config['SEQ_INFO'][k] = int(max_frame) - yaml_dataset_config['FRAME_START_IDX'] + 1
                elif k not in seqs:
                    raise FileNotFoundError('Could not find sequence {} in the dataset'.format(k))
                else:
                    print(f'sequence {k} is not None, skipping')
        else:
            print(f"'SEQ_INFO' is None, will load all sequences from {gt_structure_config['data_root']}")
            yaml_dataset_config['SEQ_INFO'] = {}
            for seq in seqs:
                gt_txt_name = gt_structure_config['gt_txt_name']
                if 'seq_name' in gt_txt_name:
                    gt_txt_name = gt_txt_name.format(seq_name=seq)
                gt_txt = gt_structure_config['gt_loc_format'].format(
                    data_root=gt_structure_config['data_root'],
                    split_name=split_name,
                    seq_name=seq,
                    gt_folder_name=gt_structure_config['gt_folder_name'],
                    gt_txt_name=gt_txt_name)
                gt_tracks = np.loadtxt(gt_txt, delimiter=',')
                max_frame = gt_tracks[:, 0].max()
                yaml_dataset_config['SEQ_INFO'][seq] = int(max_frame) - yaml_dataset_config['FRAME_START_IDX'] + 1

    # Start eval
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = CustomDataset.get_default_dataset_config()
    updated_dataset_config = CustomDataset.update_dataset_config(default_dataset_config, yaml_dataset_config)

    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **updated_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in updated_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [CustomDataset(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    combined_metrics = extract_combined_metrics(output_res)

    print(combined_metrics)
    headers = ['Class', 'HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW']
    out_put_file = os.path.join(yaml_dataset_config['tracker_structure_config']['trackers_folder'],  
                                yaml_dataset_config['tracker_structure_config']['split_name'], yaml_dataset_config['OUTPUT_SUB_FOLDER'], 'abstract.txt')
    with open(out_put_file, 'a') as f:
        f.write(','.join(map(str,  headers)) + '\n')  
    for dataset_name, dataset_metrics in combined_metrics.items():
        for path, path_metrics in dataset_metrics.items():
            for label, metrics in path_metrics.items():
                print(f'{dataset_name} {path}')
                print(f'TrackEval COMBINED Metrics (All Sequences Merged):{dataset_name}-{path} ')
                table_data = [[
                    label, f"{metrics['HOTA']:.3f}", f"{metrics['DetA']:.3f}", f"{metrics['AssA']:.3f}", f"{metrics['MOTA']:.3f}",
                    f"{metrics['IDF1']:.3f}", metrics['IDSW']]]

                print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))

                with open(out_put_file, 'a') as f:
                    f.write(','.join(
                        [
                    label, str(f"{metrics['HOTA']:.3f}"), str(f"{metrics['DetA']:.3f}"), str(f"{metrics['AssA']:.3f}"), str(f"{metrics['MOTA']:.3f}"),
                    str(f"{metrics['IDF1']:.3f}"), str(metrics['IDSW'])]
                    ) + '\n')  
    print(f"Eval results saved to: {os.path.join(yaml_dataset_config['tracker_structure_config']['trackers_folder'],yaml_dataset_config['tracker_structure_config']['split_name'], yaml_dataset_config['OUTPUT_SUB_FOLDER'])}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/template2.yaml', help='custom config file')

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        yaml_dataset_config = yaml.safe_load(f)

    main(args, yaml_dataset_config)
