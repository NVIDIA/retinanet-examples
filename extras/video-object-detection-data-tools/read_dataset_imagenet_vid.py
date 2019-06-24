import os
import xml.etree.ElementTree as ET

def read_imagenet(data_dir, data_part, existing_lens):

    assert data_part in _get_allowed_data_part(), 'Data part must be one of %s' % ", ".join(_get_allowed_data_part())
    assert os.path.isdir(data_dir), "data_dir does not exist. %s" % data_dir

    image_dir = os.path.join(data_dir, 'Data', 'VID', data_part)
    ann_dir = os.path.join(data_dir, 'Annotations', 'VID', data_part)

    # Get categories
    cat_file = os.path.join(data_dir, 'Annotations', 'categories', '200_cat_synset.txt')
    cats = _get_categories(cat_file, existing_lens['cats'])
    catsynset2id = _make_catsynset2id(cats)

    # Get videos
    vids = _get_video_names(image_dir, existing_lens['vids'])

    # Get images
    imgs = _get_images(vids, image_dir, ann_dir, existing_lens['imgs'], test_data=data_part=='test')

    # Get annotations
    anns = []
    if data_part!='test':
        anns = _get_annotations(imgs, ann_dir, existing_lens['anns'], catsynset2id)

    return cats, vids, imgs, anns



def recursively_list_video_dirs(search_dir, extension):
    """
    Recursively search for directories that themselves contain frames
    :param search_dir: Directory to search through
    :param extension: eg '.JPEG' to search for (can be list)
    :return: list of the abs path to the directories
    """
    if type(extension) == str:
        extension = [extension.upper()]
    else:
        assert type(extension) == list
        extension = [ext.upper() for ext in extension]
    found_dirs = []
    list_this_dir = os.listdir(search_dir)
    for item in list_this_dir:
        abs_path = os.path.join(search_dir, item)
        if os.path.isdir(abs_path):
            if os.path.splitext(os.listdir(abs_path)[0])[1] in extension:
                # The first file inside this directory is of the type we are looking for
                found_dirs.append(abs_path)
            else:
                if os.path.isdir(abs_path):
                    new_dirs = recursively_list_video_dirs(abs_path, extension)
                    found_dirs.extend(new_dirs)
    return found_dirs

def _get_allowed_data_part():
    return ['train', 'val', 'test']


def _get_categories(cat_file, offset):
    """Get categories from cat_file.
    Return list of categories, each item is a dict containing name, id and supercategory"""
    if not os.path.isfile(cat_file):
        raise IOError('  ImageNet: The categories file does not exist in the expected location: %s' % cat_file)
    with open(cat_file, 'r') as fid:
        lines = fid.readlines()
    cat_dict = {}
    for line in lines:
        parts = line.split(' ')
        synset = parts[0]
        name = ' '.join(parts[1:])
        cat_dict[synset] = name.strip()
    cats = [{'name': cat_dict[synset], 'id': id_ + offset, 'synset': synset, 'supercategory': 'object'} for id_, synset in
            enumerate(cat_dict)]
    return cats

def _get_video_names(vid_data_dir, offset):
    video_names = []
    vid_dirs = recursively_list_video_dirs(vid_data_dir, '.JPEG')
    for vid_dir in vid_dirs:
        video_name = os.path.relpath(vid_dir, vid_data_dir)
        video_names.append(video_name)
    return [{'name': vid_name, 'id': ii + offset} for ii, vid_name in enumerate(video_names)]

def _get_images(vids, image_dir, ann_dir, offset, test_data=False):
    image_list = []
    id_ = offset  # for assinging image ids
    for this_vid in vids:
        vid_name = this_vid['name']
        vid_id = this_vid['id']
        vid_dir = os.path.join(image_dir, vid_name)
        # Get a list of frames
        frame_list = os.listdir(vid_dir)
        frame_list.sort()
        frame_abs_list = [os.path.join(vid_dir, frame) for frame in frame_list]
        ann_abs_list = [os.path.join(ann_dir, vid_name, os.path.splitext(frame)[0] + '.xml') for frame in
                        frame_list]
        # Now loop through frames, saving them to image_list if both frame_abs and ann_abs exist.
        for ii in range(len(frame_list)):
            this_frame = frame_abs_list[ii]
            this_ann = ann_abs_list[ii]
            frame_name = os.path.splitext(frame_list[ii])[0]
            # Do they both exist?
            if os.path.isfile(this_frame) and (os.path.isfile(this_ann) or test_data):
                if not test_data:
                    try:
                        _ = ET.parse(this_ann)
                    except UserWarning:
                        continue
                # Now add this file to image_list
                this_entry = {'id': id_, 'file_name': frame_list[ii], 'file_path': this_frame,
                              'video_name': vid_name, 'video_id': vid_id,
                              'width': 0, 'height': 0, 'license': 1, 'flickr_url': "", 'coco_url': "",
                              'date_captured': ""}
                image_list.append(this_entry)
                id_ += 1
    return image_list

def _get_annotations(imgs, ann_dir, offset, catsynset2id):
    annotations_list = []

    id_ = offset  # for assigning annotation ids.

    for this_img in imgs:
        vid_name = this_img['video_name']
        vid_ann_dir = os.path.join(ann_dir, vid_name)
        file_name = os.path.splitext(this_img['file_name'])[0] + '.xml'
        vid_ann_file_path = os.path.join(vid_ann_dir, file_name)

        boxes = _imagenet_read_annotation_xml(vid_ann_file_path)

        for box in boxes:
            x, y, w, h, cat_synset = box
            cat_id = catsynset2id[cat_synset]
            this_entry = {'id': id_, 'image_id': this_img['id'], 'category_id': cat_id,
                          'segmentation': [], 'area': 0, 'iscrowd': 0,
                          'video_id': this_img['video_id'],
                          'bbox': [x, y, w, h]}
            annotations_list.append(this_entry)
            id_ += 1
    return annotations_list

def _imagenet_read_annotation_xml(annotation_filepath):
    """Read filepath, return list of boxes. Each box is list: x, y, w, h, cat_synset"""

    try:
        ann = read_xml_annotations(annotation_filepath)
    except Exception as err:
        print("ERROR: annotation %s" % annotation_filepath)
        print("ERROR:", err)

    # Create boxes
    boxes = []  # To store boxes
    for obj in ann['objects']:
        xmin = obj['bbox']['xmin']
        xmax = obj['bbox']['xmax']
        ymin = obj['bbox']['ymin']
        ymax = obj['bbox']['ymax']

        xx = xmin
        yy = ymin
        width = xmax - xmin
        height = ymax - ymin

        cat_synset = obj['obj_class']

        # Now find the class using 'semantic'
        if width > 1 and height > 1:
            boxes.append((xx, yy, width, height, cat_synset))
    return boxes

def _make_catsynset2id(cats):
    cat2id = {}
    for cat in cats:
        cat2id[cat['synset']] = cat['id']
    return cat2id

def read_xml_annotations(annotation_filename):
    """
    Reads xml from annotation file and returns dictionary
    :param annotation_filename: path to annotiation file
    :return: ann, a dict with the following attributes:
        ann['image_width']: image width in pixels.
        ann['image_height']: image hieght in pixels.
        ann['objects']: a list of objects. Each of which is a dictionary.
            ann['objects'][0]['trackid']: Integer track id.
            ann['objects'][0]['obj_class']: The nxxxxxxx style category code.
            ann['objects'][0]['occluded']: 0 or 1?????
            ann['objects'][0]['generated']: 0 or 1?????
            ann['objects'][0]['bbox']: dict containing integer pixel values for xmin, xmax, ymin, ymax
    """
    ann = {}
    tree = ET.parse(annotation_filename)
    root = tree.getroot()
    ann['image_width'] = int(root.find('size').find('width').text)
    ann['image_height'] = int(root.find('size').find('height').text)
    ann['objects'] = []  # list for storing objects
    for obj in root.findall('object'):
        this_object = {}  # dictionary for this object
        try:
            this_object['trackid'] = int(obj.find('trackid').text)
        except AttributeError:
            this_object['trackid'] = 0
        this_object['obj_class'] = obj.find('name').text
        this_object['bbox'] = {}
        for bboxpart in ['xmin', 'xmax', 'ymin', 'ymax']:
            this_object['bbox'][bboxpart] = int(obj.find('bndbox').find(bboxpart).text)
        try:
            this_object['occluded'] = int(obj.find('occluded').text)
        except AttributeError:
            this_object['occluded'] = 0
        try:
            this_object['generated'] = int(obj.find('generated').text)
        except AttributeError:
            this_object['generated'] = 0
        ann['objects'].append(this_object)
    return ann

