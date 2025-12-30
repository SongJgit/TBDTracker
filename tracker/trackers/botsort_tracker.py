"""Bot sort."""

import numpy as np
import torch

import cv2
import torchvision.transforms as T

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid
from .matching import iou_distance, fuse_det_score, embedding_distance, linear_assignment
from typing import Optional, Dict, Any

# for reid
from .reid_models.engine import load_reid_model, crop_and_resize, select_device

from .camera_motion_compensation.cmc import GMC

# base class
from .basetracker import BaseTracker


class BotTracker(BaseTracker):

    def __init__(
            self,
            init_thresh: float = 0.7,
            track_thresh_high: float = 0.6,
            track_thresh_low: float = 0.1,
            match_thresh: float = 0.8,
            proximity_thresh: float = 0.5,
            fuse_detection_score: bool = False,
            reid_cfg: Optional[Dict[str, Any]] = None,
            motion_format: Dict[str, Any] | str | None = 'botsort',
            track_buffer: int = 30,
            frame_rate: int = 30,
            cmc_cfg: Optional[Dict[str, Any]] = dict(cmc_method='orb', downscale=4),
    ) -> None:

        super().__init__(
            init_thresh=init_thresh,
            motion_format=motion_format,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )

        self.track_thresh_high = track_thresh_high
        self.track_thresh_low = track_thresh_low
        self.match_thresh = match_thresh
        self.fuse_detection_score = fuse_detection_score
        self.proximity_thresh = proximity_thresh

        self.reid_cfg = reid_cfg
        if self.reid_cfg is not None:
            self.reid_model = load_reid_model(self.reid_cfg.reid_model,
                                              self.reid_cfg.model_path,
                                              device=self.reid_cfg.device,
                                              trt=self.reid_cfg.trt,
                                              crop_size=self.reid_cfg.crop_size)

        self.gmc = GMC(method=cmc_cfg.cmc_method, downscale=cmc_cfg.downscale, verbose=None)
        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlwh format
        """

        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        remain_inds = scores > self.track_thresh_high
        inds_low = scores > self.track_thresh_low
        inds_high = scores < self.track_thresh_high

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        cates = categories[remain_inds]
        cates_second = categories[inds_second]

        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        """Step 1: Extract reid features"""
        if self.reid_cfg is not None:
            features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img, crop_size=self.reid_cfg.crop_size)

        height = ori_img.shape[0]
        width = ori_img.shape[1]

        if len(dets) > 0:
            if self.reid_cfg is not None:
                detections = [
                    Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat, img_size=[height, width])
                    for (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
            else:
                detections = [
                    Tracklet(tlwh, s, cate, motion=self.motion, img_size=[height, width])
                    for (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []
        ''' Add newly detected tracklets to tracked_tracklets'''
        unconfirmed = []
        tracked_tracklets = []  # type: list[Tracklet]
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)
        ''' Step 2: First association, with high score detection boxes'''
        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # Camera motion compensation
        if not isinstance(self.motion, dict):
            # KNet don't need camera motion compensation
            warp = self.gmc.apply(ori_img, dets)
            self.gmc.multi_gmc(tracklet_pool, warp)
            self.gmc.multi_gmc(unconfirmed, warp)

        ious_dists = iou_distance(tracklet_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)  # high conf iou

        # fuse detection conf into iou dist
        if self.fuse_detection_score:
            ious_dists = fuse_det_score(ious_dists, detections)

        if self.reid_cfg is not None:
            # mixed cost matrix
            emb_dists = embedding_distance(tracklet_pool, detections) / 2.0
            emb_dists[emb_dists > 0.25] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections."""
            detections_second = [
                Tracklet(tlwh, s, cate, motion=self.motion, img_size=[height, width])
                for (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_tracklets, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = r_tracked_tracklets[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # fuse detection conf into iou dist
        if self.fuse_detection_score:
            ious_dists = fuse_det_score(ious_dists, detections)

        if self.reid_cfg is not None:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > 0.25] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)
        """ Step 4: Init new tracklets"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets
