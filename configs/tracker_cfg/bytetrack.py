from tracker import ByteTracker

tracker = dict(
    type=ByteTracker,
    # reid_cfg = dict(
    # reid_model='osnet_x0_25',
    # model_path='./weights/osnet_x0_25.pth',
    # trt = False,
    # crop_size=[128, 64],
    # device='cuda:0',
    # ),
    frame_rate=30,
    init_thresh=0.6,
    track_thresh_high=0.6,
    track_thresh_low=0.1,
    match_thresh=0.9,
    motion_format='byte',
    track_buffer=30,
    fuse_detection_score=False,
)
