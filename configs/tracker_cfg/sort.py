from tracker import SortTracker

tracker = dict(
    type=SortTracker,
    frame_rate=30,
    init_thresh=0.6,
    track_thresh=0.5,
    match_thresh=0.7,
    motion_format='sort',
    track_buffer=30,
)
