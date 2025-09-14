from .trackers.byte_tracker import ByteTracker
from .trackers.sort_tracker import SortTracker
from .trackers.botsort_tracker import BotTracker
from .trackers.c_biou_tracker import C_BIoUTracker
from .trackers.ocsort_tracker import OCSortTracker
from .trackers.deepsort_tracker import DeepSortTracker
from .trackers.strongsort_tracker import StrongSortTracker
from .trackers.sparse_tracker import SparseTracker
from .trackers.ucmc_tracker import UCMCTracker
from .trackers.hybridsort_tracker import HybridSortTracker
from .trackers.tracktrack_tracker import TrackTrackTracker
from .trackers.improassoc_tracker import ImproAssocTracker

TRACKER_DICT = {
    'sort': SortTracker,
    'bytetrack': ByteTracker,
    'botsort': BotTracker,
    'c_bioutrack': C_BIoUTracker,
    'ocsort': OCSortTracker,
    'deepsort': DeepSortTracker,
    'strongsort': StrongSortTracker,
    'sparsetrack': SparseTracker,
    'ucmctrack': UCMCTracker,
    'hybridsort': HybridSortTracker}
