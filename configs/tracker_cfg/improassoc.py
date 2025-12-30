from tracker import ImproAssocTracker

tracker = dict(
    type=ImproAssocTracker,
    # reid_cfg = dict(
    # reid_model='osnet_x0_25',
    # model_path='./weights/osnet_x0_25.pth',
    # trt = False,
    # crop_size=[128, 64],
    # device='cuda:0'),
    init_thresh=0.5,
    track_thresh_low=0.1,
    track_thresh_high=0.5,
    match_thresh=0.65,  # default _d_h_max in paper.
    second_match_thresh=0.19,  # default _d_l_max in paper.
    lambda_=0.05,
    proximity_thresh=0.1,  # default _o_min in paper.
    overlap_thresh=0.55,  # default _o_max in paper.
    motion_format='improassoc',
    track_buffer=35,
    frame_rate=30,
    cmc_cfg=dict(cmc_method='orb', downscale=2),
)
