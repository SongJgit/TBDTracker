import cv2
import os
import numpy as np
from PIL import Image
from mmengine.visualization import Visualizer

def plot_img(img, frame_id, results, save_dir):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert img is not None

    if len(img.shape) > 3:
        img = img.squeeze(0)

    img_ = np.ascontiguousarray(np.copy(img))
    
    tlwhs, ids, clses = results[0], results[1], results[2]

    for tlwh, id, cls in zip(tlwhs, ids, clses):

        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(
            img_,
            tlbr[:2],
            tlbr[2:],
            get_color(id),
            thickness=3,
        )
        # note the id and cls
        # text = f'{int(cls)}_{id}'
        text = f'{id}'
        cv2.putText(img_,
                    text, (tlbr[0]-2, tlbr[1]),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    
                    color=(255, 164, 0),
                    thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)


def get_color(idx):
    """aux func for plot_seq get a unique color for each id."""
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_img_vis(img, frame_id, results, save_dir):
    assert img is not None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_ = np.ascontiguousarray(np.copy(img))[:, :, ::-1]
    vis = Visualizer(image=img_, fig_save_cfg=dict(frameon=False, dpi=300))

    if len(results[1])!=0:
        tlwhs, ids, clses = results[0], results[1], results[2]    
        tlwhs = np.array(tlwhs) # [n, 4]
        tlbrs = np.copy(tlwhs)
        tlbrs[:, [2,3]] = tlwhs[:, [2,3]] + tlwhs[:, [0,1]]
        tlbrs[:, 2] = np.clip(tlbrs[:, 2], 0, img_.shape[1]-1)
        tlbrs[:, 3] = np.clip(tlbrs[:, 3], 0, img_.shape[0]-1)

        ids = np.array(ids).flatten()
        colors = np.vstack(get_color(ids)).T # [n,3]

        colors = [get_color(id) for id in ids]
        vis = Visualizer(image=img_, fig_save_cfg=dict(frameon=False, dpi=300))
        vis.draw_bboxes(
            bboxes = tlbrs, edge_colors=colors, line_widths=1
        ).draw_texts(ids.tolist(), positions=tlbrs[:, [0,1]],vertical_alignments='bottom', font_families='Arial', font_sizes=8, colors=(255, 164, 0))
    
    # img = vis.get_image()
    # cv2.imwrite('../forpaper/'+video_name+'/'+'{:08d}.png'.format(frame_id), img[:, :, ::-1])
    vis.fig_save.savefig(os.path.join(save_dir, f'{frame_id:05d}.jpg'))
    


def save_video(save_path, images_path):
    """save images (frames) to a video."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    images_list = sorted(os.listdir(images_path))
    # save_video_path = os.path.join(save_path, 'videos', seq + '.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    img0 = Image.open(os.path.join(images_path, images_list[0]))
    vw = cv2.VideoWriter(save_path, fourcc, 15, img0.size)

    for image_name in images_list:
        image = cv2.imread(filename=os.path.join(images_path, image_name))
        vw.write(image)
