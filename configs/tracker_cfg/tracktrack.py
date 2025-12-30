from tracker import TrackTrackTracker
# TODO
tracker = dict(
    type=TrackTrackTracker,
    reid_cfg=dict(reid_model='osnet_x0_25',
                  model_path='./weights/osnet_x0_25.pth',
                  trt=False,
                  crop_size=[128, 64],
                  device='cuda:0'),
    init_thresh=0.6,
    track_thresh_high=0.6,
    track_thresh_low=0.1,
    tail_thresh=0.55,
    delta_t=3,
    penalty_p=0.2,
    penalty_q=0.4,
    reduce_step=0.05,
    match_thresh=0.8,
    motion_format='tracktrack',
    cmc_cfg=dict(cmc_method='orb', downscale=2),
    track_buffer=30,
    frame_rate=30,
)
