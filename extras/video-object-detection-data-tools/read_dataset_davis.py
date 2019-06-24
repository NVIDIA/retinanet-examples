import os
import json
from PIL import Image
import numpy as np


def read_davis(data_dir, data_part, existing_lens, year=2017, data_res='full'):
    """

    :param data_dir: Directory containing DAVIS data
    :param data_part:
    :param existing_lens: Dictionary of offsets to add to the indexes
    :param year:
    :param data_res:
    :return: cats, vids, imgs, anns
    """
    assert data_part in _allowed_data_part(year), 'Data part must be one of %s' % ", ".join(_allowed_data_part(year))
    assert os.path.isdir(data_dir), "data_dir does not exist. %s" % data_dir
    assert data_res in ['480p', 'full'], "data_res must be 480p or full"
    if data_res == 'full':
        data_res = 'Full-Resolution'

    # Get categories, as defined by DAVIS.
    cats, cat_original_to_new = _get_categories(os.path.join(data_dir, 'categories.json'), existing_lens['cats'])

    # Get list of videos for this data type
    image_set_path = os.path.join(data_dir, 'ImageSets', str(year), data_part + '.txt')
    video_names = _read_davis_ImageSet(image_set_path)
    vids = _get_videos(video_names, existing_lens['vids'])

    # Get list of images, indexing the video by video_id
    image_dir = os.path.join(data_dir, 'JPEGImages', data_res)
    imgs = _get_images(vids, image_dir, existing_lens['imgs'])

    # Get annotations
    ann_dir = os.path.join(data_dir, 'Annotations', data_res)
    ann_semantic_dir = os.path.join(data_dir, 'Annotations_semantics', data_res)
    anns = _get_annotations(imgs, ann_dir, ann_semantic_dir, existing_lens['anns'], cat_original_to_new)
    return cats, vids, imgs, anns


def _allowed_data_part(year):
    if year == 2017:
        return ['train', 'val', 'test-dev', 'test-challenge']
    else:
        return ['train', 'val']


def _get_categories(cat_file, offset):
    """Get categories from cat_file.
    Return list of categories, each item is a dict containing name, id and supercategory."""
    with open(cat_file) as f:
        catdata = json.load(f)

    cat_original_to_new = {}
    cats = []
    for cat in catdata:
        cat_dict = {}
        cat_dict['name'] = cat
        cat_dict['id'] = int(catdata[cat]['id']) + offset
        cat_dict['supercategory'] = catdata[cat]['super_category']
        cat_original_to_new[int(catdata[cat]['id'])] = int(catdata[cat]['id']) + offset
        cats.append(cat_dict)
    return cats, cat_original_to_new
    # return [{'name': cat, 'id': int(catdata[cat]['id']) + offset, 'supercategory': catdata[cat]['super_category']} for cat in
    #         catdata]


def _read_davis_ImageSet(imageset_file):
    """Return list of videos"""
    with open(imageset_file, 'r') as fid:
        image_set_raw = fid.readlines()
    image_set = [line.strip() for line in image_set_raw]
    return image_set


def _get_videos(image_set, offset):
    """Return a list of videos, each item is a dict containing name, id"""
    # Get list of videos
    return [{'name': vid_name, 'id': ii + offset} for ii, vid_name in enumerate(image_set)]


def _get_images(vid_list, image_dir, offset):
    """
    Return list of iamges, each item is a dict containing:
    id              int
    file_name		str 	image file name
    file_path		str	    full path to image
    video_name		str	    name of video
    video_id		int
    """
    image_list = []

    id_ = offset  # for assinging image ids

    for this_vid in vid_list:
        vid_name = this_vid['name']
        vid_id = this_vid['id']
        vid_image_dir = os.path.join(image_dir, vid_name)

        # Get list of directory contents
        vid_frame_list_raw = os.listdir(vid_image_dir)
        vid_frame_list_raw.sort()
        vid_frame_list = [vid_frame for vid_frame in vid_frame_list_raw if
                          os.path.splitext(vid_frame)[1] == '.jpg']
        for vid_frame in vid_frame_list:
            # if vid_name not in sing_vids or '00000' in vid_frame:
            this_entry = {'id': id_, 'file_name': vid_frame, 'file_path': os.path.join(vid_image_dir, vid_frame),
                          'video_name': vid_name, 'video_id': vid_id,
                          'width': 0, 'height': 0, 'license': 1, 'flickr_url': "", 'coco_url': "",
                          'date_captured': ""}
            image_list.append(this_entry)
            id_ += 1
    return image_list


def _get_annotations(image_list, ann_dir, ann_semantic_dir, offset, cat_original_to_new):
    """ Return list of annotations, each item as a dict containing:
	id			    int
	image_id		int
	category_id		int
	video_id		int
	bbox			4 x float	[x, y, w, h]
    """
    annotations_list = []

    id_ = offset  # for assigning annotation ids.

    for this_img in image_list:
        vid_name = this_img['video_name']
        vid_ann_dir = os.path.join(ann_dir, vid_name)
        vid_ann_semantic_dir = os.path.join(ann_semantic_dir, vid_name)
        file_name = os.path.splitext(this_img['file_name'])[0] + '.png'
        vid_ann_file_path = os.path.join(vid_ann_dir, file_name)
        vid_ann_semantic_file_path = os.path.join(vid_ann_semantic_dir, file_name)
        boxes = _read_annotation_png(vid_ann_file_path, vid_ann_semantic_file_path)

        for box in boxes:
            x, y, w, h, cat_id, imgwidth = box
            this_entry = {'id': id_, 'image_id': this_img['id'], 'category_id': cat_original_to_new[cat_id],
                          'segmentation': [], 'area': 0, 'iscrowd':0,
                          'video_id': this_img['video_id'],
                          'bbox': [x, y, w, h]}
            if vid_name != 'color-run' or w < imgwidth:
                annotations_list.append(this_entry)
                id_ += 1
    return annotations_list


def _read_annotation_png(annotation_filepath, semantic_filepath):
    """Read filepath, return list of boxes. Each box is list: x, y, w, h, raw_class, imgwidth"""
    ann_pil = Image.open(annotation_filepath)
    ann = np.array(ann_pil)

    imgwidth = ann.shape[0]

    semantic_pil = Image.open(semantic_filepath)
    semantic = np.array(semantic_pil)

    # Get unique values (instances IDs)
    vals = np.unique(ann)
    instances = vals[vals > 0]

    # Create boxes
    boxes = []  # To store boxes
    for indx in instances:
        # Get all coords of these pixels
        coords = np.where(ann == indx)
        try:
            xmin = int(np.min(coords[1]))
            ymin = int(np.min(coords[0]))
            xmax = int(np.max(coords[1]))
            ymax = int(np.max(coords[0]))
        except ValueError:
            continue

        xx = xmin
        yy = ymin
        width = xmax - xmin
        height = ymax - ymin

        # Now find the class using 'semantic'
        class_votes = semantic[ann == indx]
        class_ = int(np.bincount(class_votes).argmax())
        if class_ == 0:
            # This happens in some unfortunate edge cases, when the object is partially obscured.
            votes = np.bincount(class_votes)
            # We know argmax is zero
            try:
                class_ = votes[1:].argmax() + 1 # Exclude zero as an answer
            except ValueError:
                if 'tennis' in annotation_filepath:
                    # Assume tennis ball
                    class_ = 3
                else:
                    class_ = 0
        if width > 1 and height > 1 and class_ != 0:
            ## DEBUG SECTION
            boxes.append((xx, yy, width, height, class_, imgwidth))
    return boxes