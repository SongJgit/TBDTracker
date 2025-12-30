from tracker import SparseTracker

tracker = dict(
    type=SparseTracker,
    init_thresh=0.65,  # det_thresh, common is track_thresh_high+0.05
    track_thresh_high=0.6,
    track_thresh_low=0.1,
    match_thresh=0.75,
    confirm_thresh=0.8,
    motion_format='sparse',
    track_buffer=30,
    frame_rate=30,
    depth_levels_high=1,
    depth_levels_low=3,
    cmc_cfg=dict(cmc_method='orb', downscale=4),
)
