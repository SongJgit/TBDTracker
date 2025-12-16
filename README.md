<h1 align="center">TBDTracker: Tracking-by-Detection Trackers for Multiple Object Tracking</h1>
<p align="center">
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit" style="max-width:100%;"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
  <a href=""><img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-v1.8+-E97040?logo=pytorch&logoColor=white"></a>
</p>

## Notice

We will upload all the code once the paper has been accepted.


## 📄 Documentation


<details open>
<summary>Install</summary>

If you want to use Learning-aided Kalman filtering (LAKF), you need to install FilterNet [FilterNet](https://github.com/SongJgit/filternet). 

Otherwise, you only need to install the required dependencies:

```bash
pip install -e .
```

</details>

<details open>
<summary>Usage</summary>

For convenience, we have separated detection and tracking. This allows detection to run only once, enabling different trackers to reuse the same detection results. This approach saves inference time for detectors while ensuring consistency, making it easier for those focused solely on tracker performance.
Of course, detection, tracking, and evaluation can also be completed in a single step.
 Consequently, the approach can be categorized as follows:
1. Detection + Tracking + Evaluation
2. Detection + Tracking, Evaluation
3. Detection, Tracking + Evaluation

We consider the third approach to be the best practice: running detection independently while simultaneously running tracking and evaluation.

</details>

### Preprocess

#### Convert the dataset format to the required format
We default to using the YOLO format, so we need to convert the standard MOT format to YOLO format.

Convert the train dataset to YOLO format:
```bash
 python tools/convert_StandardMOT_to_yolo.py --data_root /data/MOT_Datasets/DanceTrack --split train --generate_imgs
```

Convert the val dataset to YOLO format:
```bash
 python tools/convert_StandardMOT_to_yolo.py --data_root /data/MOT_Datasets/DanceTrack --split val --generate_imgs
```

If half the train dataset:
```bash
python tools/half_dataset.py --source /data/MOT_Datasets/DanceTrack/train copy
```
The half-dataset is stored in `/data/MOT_Datasets/DanceTrack-tr50-val50`. 
Then perform the preceding format conversion, but note that `--data_root` needs to be modified to `/data/MOT_Datasets/DanceTrack-tr50-val50`.


### ✨ Best Practices: Detection, Tracking + Evaluation
#### 1. Detect
Modify the `DATASET_ROOT` parameter in `./cfg/data_cfg/dancetrack-val.yaml` to your local directory path.

Then run
```bash
sh ./scripts/dance_detect.sh
```
Detection results are stored in `./results_detected/dance/val`.

#### 2. Track + Evaluation
Modify the `data_root` parameter in `./cfg/eval_cfg/DanceTrack.yaml` to your local directory path (same to `DATASET_ROOT` in `./cfg/data_cfg/dancetrack-val.yaml`).

Modify the `DETECTED_FOLDER` parameter in `./scripts/track_from_fold_dance_sort_kf.sh` to detection results path `./results_detected/dance/val`.

Then run
```bash
sh ./scripts/track_from_fold_dance_sort_kf.sh
```

You will receive the following outputs

```bash
CustomDataset ./results_tracked/dancetrack-val/sort/fromdet-KF/val
TrackEval COMBINED Metrics (All Sequences Merged):CustomDataset-./results_tracked/dancetrack-val/sort/fromdet-KF/val
+---------+--------+--------+--------+--------+--------+--------+
|  Class  |   HOTA |   DetA |   AssA |   MOTA |   IDF1 |   IDSW |
+=========+========+========+========+========+========+========+
|  valid  | 53.072 | 76.447 | 36.998 | 91.964 | 56.685 |   1693 |
+---------+--------+--------+--------+--------+--------+--------+
Eval results saved to: ./results_tracked/dancetrack-val/sort/fromdet-KF/val/trackeval
Evaluation done
Tracked results saved to ./results_tracked/dancetrack-val/sort/fromdet-KF/val
Eval track from ./results_tracked/dancetrack-val/sort/fromdet-KF/val
```

Tracking results and evaluation results are stored in `./results_tracked/dancetrack-val/sort/fromdet_KF`.



### Only Evaluation
Modify the parameters in `./cfg/eval_cfg/DanceTrack.yaml`:
```
data_root: Same as `DATASET_ROOT` in `./cfg/data_cfg/dancetrack-val.yaml`
trackers_folder: The tracked results folder like  `./results_tracked/dancetrack-val/sort/fromdet_KF`
```

Then run 
```bash
sh ./scripts/eval.sh
```
The evaluation results are stored in  the tracked results folder  `./results_tracked/dancetrack-val/sort/fromdet_KF/trackeval`


## Benchmark

**Notes**:The learning-aided Kalman filter (LAKF), such as KalmanNet(KNet), Split-KalmanNet(SKNet), and Semantic-Independent-KalmanNet(SIKNet), were trained on a semi-synthetic dataset, then integrated into the Tracker as an replacement to KF. Therefore, the noise parameters in the semi-synthetic dataset affect performance. Consequently, different versions of HOTA exist, as they originate from two distinct papers.


### DanceTrack
NOTES: The evaluation was conducted on the DanceTrack validation set. Oracle detections.
The semi-synthetic dataset used to train LAKF was constructed from the first half of the trajectories in the DanceTrack training set.

<div align="center">

<!-- START TRACKER TABLE -->

| Tracker | Status |Motion Model| HOTA↑ |AssA↑| MOTA↑ | IDF1↑ | IDSW|
| :-----: | :-----: | :---: | :---: | :---: |:---: |:---: |:---:|
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | | | | |||
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | | | || ||
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | KF| 49.95 | 34.80|90.41 |56.22 |1738|
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | KNet| 54.60 | 38.88|92.10 |56.22|1738|
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | SKNet| 50.94 | 35.74|89.99|54.88 |1619|
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | SIKNet| 56.19 |39.97 |92.37 |57.87|1427|
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ || | | |||
| [imprassoc](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) | ✅ || | | |||
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ || | | |||
| [hybridsort](https://arxiv.org/abs/2308.00783) | ❌ || | | |||

<!-- END TRACKER TABLE -->
</div>
</details>


<div align="center">
<table style="border-collapse: collapse; border: none; border-spacing: 0; text-align: center;">
  <!-- 表头行（加粗 + 自然行高） -->
  <tr>
    <td style="padding: 3pt 3pt;"><b>Tracker</b></td>
    <td style="padding: 3pt 3pt;"><b>Motion Model</b></td>
    <td style="padding: 3pt 3pt;"><b>Status</b></td>
    <td style="padding: 3pt 3pt;"><b>HOTA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>DetA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>AssA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>MOTA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>IDF1</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>IDSw↓</b></td>
  </tr>

  <!-- SORT 组 -->
  <tr>
    <td rowspan="5" style="padding: 3pt 3pt;"><a href="http://ieeexplore.ieee.org/document/7533003/" target="_blank">SORT</a></td>
    <td style="padding: 3pt 3pt;">Kalman filter (Original)</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">53.07</td>
    <td style="padding: 3pt 3pt;">76.45</td>
    <td style="padding: 3pt 3pt;">37.00</td>
    <td style="padding: 3pt 3pt;">91.96</td>
    <td style="padding: 3pt 3pt;">56.69</td>
    <td style="padding: 3pt 3pt;">1693</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">KNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">55.92</td>
    <td style="padding: 3pt 3pt;">79.29</td>
    <td style="padding: 3pt 3pt;">39.61</td>
    <td style="padding: 3pt 3pt;">91.90</td>
    <td style="padding: 3pt 3pt;">57.19</td>
    <td style="padding: 3pt 3pt;">1677</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SKNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">55.38</td>
    <td style="padding: 3pt 3pt;">79.27</td>
    <td style="padding: 3pt 3pt;">38.87</td>
    <td style="padding: 3pt 3pt;">91.91</td>
    <td style="padding: 3pt 3pt;">55.84</td>
    <td style="padding: 3pt 3pt;">1670</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SIKNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">57.88</td>
    <td style="padding: 3pt 3pt;">81.00</td>
    <td style="padding: 3pt 3pt;">41.52</td>
    <td style="padding: 3pt 3pt;">92.00</td>
    <td style="padding: 3pt 3pt;">57.35</td>
    <td style="padding: 3pt 3pt;">1680</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">PKNet</td>
    <td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td>
  </tr>

  <!-- ByteTrack 组 -->
  <tr>
    <td rowspan="5" style="padding: 3pt 3pt;"><a href="https://link.springer.com/10.1007/978-3-031-20047-2_1" target="_blank">ByteTrack</a></td>
    <td style="padding: 3pt 3pt;">Kalman filter (Original)</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">49.95</td>
    <td style="padding: 3pt 3pt;">71.95</td>
    <td style="padding: 3pt 3pt;">34.80</td>
    <td style="padding: 3pt 3pt;">90.41</td>
    <td style="padding: 3pt 3pt;">56.22</td>
    <td style="padding: 3pt 3pt;">1738</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">KNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">56.04</td>
    <td style="padding: 3pt 3pt;">78.56</td>
    <td style="padding: 3pt 3pt;">40.14</td>
    <td style="padding: 3pt 3pt;">92.27</td>
    <td style="padding: 3pt 3pt;">59.76</td>
    <td style="padding: 3pt 3pt;">1424</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SKNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">56.15</td>
    <td style="padding: 3pt 3pt;">78.29</td>
    <td style="padding: 3pt 3pt;">40.45</td>
    <td style="padding: 3pt 3pt;">92.06</td>
    <td style="padding: 3pt 3pt;">58.69</td>
    <td style="padding: 3pt 3pt;">1485</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SIKNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td>
    <td style="padding: 3pt 3pt;">58.42</td>
    <td style="padding: 3pt 3pt;">79.62</td>
    <td style="padding: 3pt 3pt;">43.03</td>
    <td style="padding: 3pt 3pt;">92.24</td>
    <td style="padding: 3pt 3pt;">59.67</td>
    <td style="padding: 3pt 3pt;">1452</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">PKNet</td>
    <td style="padding: 3pt 3pt;"><span style="color:#1f2328">✅</span></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td>
  </tr>
    <!-- OC-SORT 行 -->
  <tr>
    <td style="padding: 3pt 3pt;"><a href="https://arxiv.org/abs/2203.14360" target="_blank">OC-SORT</a></td>
    <td style="padding: 3pt 3pt;">Kalman filter (Original)</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">52.04</td>
    <td style="padding: 3pt 3pt;">80.55</td>
    <td style="padding: 3pt 3pt;">33.76</td>
    <td style="padding: 3pt 3pt;">91.70</td>
    <td style="padding: 3pt 3pt;">51.53</td>
    <td style="padding: 3pt 3pt;">2308</td>
  </tr>
</table>
</div>



### SoccerNet
NOTES: The evaluation was conducted on the SoccerNet testing set. Oracle detections. The semi-synthetic dataset used to train LAKF was constructed from the first half of the trajectories in the SoccerNet training set.

<div align="center">

<!-- START TRACKER TABLE -->

| Tracker | Status |Motion Model| HOTA↑ |AssA ↑| MOTA↑ | IDF1↑ | IDSW|
| :-----: | :-----: | :---: | :---: | :---: |:---: |:---: |:---:|
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | | | | |||
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | | | || ||
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | KF| 72.30 | 62.48|94.62|75.58 | 5054|
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | KNet| 75.49 | 66.67|94.71|77.71 | 4354|
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | SKNet| 73.82 | 65.14|94.24|77.43 | 4171|
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | SIKNet| 76.17 | 67.45|95.43|77.81 | 3844|
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ || | | |||
| [imprassoc](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) | ✅ || | | |||
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ || | | |||
| [hybridsort](https://arxiv.org/abs/2308.00783) | ❌ || | | |||

<!-- END TRACKER TABLE -->



</div>
</details>
<div align="center">
<table style="border-collapse: collapse; border: none; border-spacing: 0; text-align: center;">
  <tr>
    <td style="padding: 3pt 3pt;"><b>Tracker</b></td>
    <td style="padding: 3pt 3pt;"><b>Motion Model</b></td>
    <td style="padding: 3pt 3pt;"><b>Status</b></td>
    <td style="padding: 3pt 3pt;"><b>HOTA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>DetA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>AssA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>MOTA</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>IDF1</b><span style="color:#1f2328">↑</span></td>
    <td style="padding: 3pt 3pt;"><b>IDSw↓</b></td>
  </tr>

  <!-- SORT 组 -->
  <tr>
    <td rowspan="5" style="padding: 3pt 3pt;"><a href="https://ieeexplore.ieee.org/document/7533003" target="_blank">SORT</a></td>
    <td style="padding: 3pt 3pt;">Kalman filter (Original)</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">71.65</td>
    <td style="padding: 3pt 3pt;">86.78</td>
    <td style="padding: 3pt 3pt;">59.26</td>
    <td style="padding: 3pt 3pt;">93.40</td>
    <td style="padding: 3pt 3pt;">70.91</td>
    <td style="padding: 3pt 3pt;">9294</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">KNet</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">76.83</td>
    <td style="padding: 3pt 3pt;">88.41</td>
    <td style="padding: 3pt 3pt;">66.85</td>
    <td style="padding: 3pt 3pt;">94.15</td>
    <td style="padding: 3pt 3pt;">76.02</td>
    <td style="padding: 3pt 3pt;">7418</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SKNet</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">74.78</td>
    <td style="padding: 3pt 3pt;">90.07</td>
    <td style="padding: 3pt 3pt;">62.14</td>
    <td style="padding: 3pt 3pt;">94.28</td>
    <td style="padding: 3pt 3pt;">72.12</td>
    <td style="padding: 3pt 3pt;">7147</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SIKNet</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">79.02</td>
    <td style="padding: 3pt 3pt;">91.07</td>
    <td style="padding: 3pt 3pt;">68.61</td>
    <td style="padding: 3pt 3pt;">94.61</td>
    <td style="padding: 3pt 3pt;">76.70</td>
    <td style="padding: 3pt 3pt;">6270</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">PKNet</td>
    <td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td>
  </tr>

  <!-- ByteTrack 组 -->
  <tr>
    <td rowspan="5" style="padding: 3pt 3pt;"><a href="https://link.springer.com/10.1007/978-3-031-20047-2_1" target="_blank">ByteTrack</a></td>
    <td style="padding: 3pt 3pt;">Kalman filter (Original)</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">72.30</td>
    <td style="padding: 3pt 3pt;">83.44</td>
    <td style="padding: 3pt 3pt;">62.48</td>
    <td style="padding: 3pt 3pt;">94.62</td>
    <td style="padding: 3pt 3pt;">75.58</td>
    <td style="padding: 3pt 3pt;">5054</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">KNet</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">78.39</td>
    <td style="padding: 3pt 3pt;">88.49</td>
    <td style="padding: 3pt 3pt;">69.54</td>
    <td style="padding: 3pt 3pt;">95.16</td>
    <td style="padding: 3pt 3pt;">78.54</td>
    <td style="padding: 3pt 3pt;">3902</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SKNet</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">77.90</td>
    <td style="padding: 3pt 3pt;">90.57</td>
    <td style="padding: 3pt 3pt;">67.05</td>
    <td style="padding: 3pt 3pt;">95.92</td>
    <td style="padding: 3pt 3pt;">76.52</td>
    <td style="padding: 3pt 3pt;">3633</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">SIKNet</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">80.83</td>
    <td style="padding: 3pt 3pt;">91.51</td>
    <td style="padding: 3pt 3pt;">71.44</td>
    <td style="padding: 3pt 3pt;">95.82</td>
    <td style="padding: 3pt 3pt;">79.05</td>
    <td style="padding: 3pt 3pt;">3378</td>
  </tr>
  <tr>
    <td style="padding: 3pt 3pt;">PKNet</td>
    <td style="padding: 3pt 3pt;">✅</td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td><td style="padding: 3pt 3pt;"></td>
  </tr>

  <!-- OC-SORT 行 -->
  <tr>
    <td style="padding: 3pt 3pt;"><a href="https://arxiv.org/abs/2203.14360" target="_blank">OC-SORT</a></td>
    <td style="padding: 3pt 3pt;">Kalman filter (Original)</td>
    <td style="padding: 3pt 3pt;">✅</td>
    <td style="padding: 3pt 3pt;">70.07</td>
    <td style="padding: 3pt 3pt;">93.10</td>
    <td style="padding: 3pt 3pt;">52.73</td>
    <td style="padding: 3pt 3pt;">90.67</td>
    <td style="padding: 3pt 3pt;">62.94</td>
    <td style="padding: 3pt 3pt;">15405</td>
  </tr>
</table>
</div>



## Citation

If you find this repo useful, please cite our papers.
<a id="anchor1"></a>

```bibtex
@misc{song2025motionestimationmultiobjecttracking,
      title={Motion Estimation for Multi-Object Tracking using KalmanNet with Semantic-Independent Encoding},
      author={Jian Song and Wei Mei and Yunfeng Xu and Qiang Fu and Renke Kou and Lina Bu and Yucheng Long},
      year={2025},
      eprint={2509.11323},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.11323},
}
```
## Acknowledgement

The structure of this repository and much of the code is thanks to the authors of the following repositories.

- [Yolov7-tracker](https://github.com/JackWoo0831/Yolov7-tracker) : A simple multi-object tracker based on YOLO.
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) : A multi-object tracker for computer vision.















