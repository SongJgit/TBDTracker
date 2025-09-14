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

## Installation

Learning-aided Kalman filtering Trackers are implemented based on [FilterNet](https://github.com/SongJgit/filternet).

## Benchmark



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

<sub> NOTES: The evaluation was conducted on the DanceTrack validation set. Oracle detections. </sub>

</div>
</details>

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

<sub> NOTES: The evaluation was conducted on the SoccerNet testing set. Oracle detections. The official detections and embeddings used. Each tracker was configured with the original parameters provided in their official repositories. </sub>

</div>
</details>

## Acknowledgement

The structure of this repository and much of the code is thanks to the authors of the following repositories.

- [Yolov7-tracker](https://github.com/JackWoo0831/Yolov7-tracker) : A simple multi-object tracker based on YOLO.
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) : A multi-object tracker for computer vision.
