from tracker import OCSortTracker

tracker = dict(
    type=OCSortTracker,
    # reid_cfg = dict(
    # reid_model='osnet_x0_25',
    # model_path='./weights/osnet_x0_25.pth',
    # trt = False,
    # crop_size=[128, 64],
    # device='cuda:0'),
    init_thresh=0.6,
    track_thresh_high=0.6,
    track_thresh_low=0.1,
    delta_t=3,
    match_thresh=0.3,  # asso iou_thresh hold
    motion_format='ocsort',
    track_buffer=30,
    frame_rate=30,
)
