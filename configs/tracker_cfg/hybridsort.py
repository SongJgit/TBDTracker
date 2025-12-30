from tracker import HybridSortTracker

tracker = dict(
    type=HybridSortTracker,
    init_thresh=0.6,
    track_thresh_high=0.12442660055370669,
    track_thresh_low=0.1,
    delta_t=5,
    match_thresh=0.3,  # asso iou_thresh hold
    motion_format='hybridsort',
    track_buffer=30,
    frame_rate=30,
)
