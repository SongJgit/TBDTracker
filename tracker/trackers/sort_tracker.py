"""Sort."""

from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet
from .matching import linear_assignment, iou_distance
# base class
from .basetracker import BaseTracker
from typing import Optional, Any, Dict


class SortTracker(BaseTracker):

    def __init__(
        self,
        init_thresh: float = 0.6,
        track_thresh: float = 0.5,
        match_thresh: float = 0.7,
        motion_format: Dict[str, Any] | str | None = 'sort',
        track_buffer: int = 30,
        frame_rate: int = 30,
    ) -> None:

        super().__init__(
            init_thresh=init_thresh,
            motion_format=motion_format,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )

        self.match_thresh = match_thresh
        self.track_thresh = track_thresh

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlbr format
        """

        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        remain_inds = scores > self.track_thresh

        dets = bboxes[remain_inds]

        cates = categories[remain_inds]

        scores_keep = scores[remain_inds]
        height = ori_img.shape[0]
        width = ori_img.shape[1]

        if len(dets) > 0:
            """Detections."""
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

        dists = iou_distance(tracklet_pool, detections)

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

        for it in u_track:
            track = tracklet_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)
        """ Step 3: Init new tracklets"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)
        """ Step 4: Update state"""
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets
