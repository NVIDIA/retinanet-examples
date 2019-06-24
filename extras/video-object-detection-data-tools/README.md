# Video Object Detection - Data Tools

**Purpose:** These tools have been created to enable users to convert video object detection datasets into a common
 (ODIMI) format, to minimise the time-to-value for training RetinaNet and other detection networks that expect annotations in 
 the COCO format.

The ODIMI format is explained below.

## Quick start guide
**Pre-requisists**

* Python 3
* Python modules os, json, datetime, numpy, random, warnings, xml and PIL
* Python cv2, for creating video outputs 
  * pip install opencv-python

**Converting the DAVIS 2017 dataset to ODIMI format**

We assume that you have downloaded and extracted the DAVIS data to `/data/DAVIS`

```python
import prepare_video_datasets as pvd
davis_train = pvd.OdimiDataset(version="DAVIS original training", description="The complete DAVIS train dataset.")
# Load DAVIS raw TRAIN data
davis_train.load_raw_data('davis', 'train', '/data/DAVIS')
# Reset numbers, so that the cateogry IDs run from 0 to 77
davis_train.reset_catids()
# Write DAVIS data
davis_train.write_data('/data/DAVIS/json/DAVIS-2017-original_train.json')
```
Similarly for validation data:
```python
davis_val = pvd.OdimiDataset(version="DAVIS original validation", description="The complete DAVIS validation dataset.")
davis_val.load_raw_data('davis', 'val', '/data/DAVIS')
davis_val.reset_catids()
davis_val.write_data('/data/DAVIS/json/DAVIS-2017-original_val.json')

```
## Methods

**OdimiDataset(version="", description="")**: Create a dataset instance, with version (str) and description (str).

**load_raw_data(data_style, data_part, data_dir)**: Loads raw data.
* data_style: str [*davis* | *imagenet_vid*] The original source of the data.
* data_part: str. The train/val/test part of the data. The accepted terms vary with the data style.
* data_dir: str. The path to the data.

**load_json_data(filepath)**: Loads JSON dataset file, as produced by the `write_data` method of this code.

**explore_data(aspect, catslist=None)**: Prints a representation of the data
* `aspect = 'all'`: Equivalent to method `summary`
* `aspect = 'cats`: Prints a list of categories, and counts the number of annotations in each category.
* `aspect = 'vids_with_cats`: Lists videos (IDs and names) that have at least one annotation in one of the classes supplied as catslist.
* `aspect = 'vids_only_with_cats'`: Lists videos (IDs and names) where all of the annotations are in catslist.

**remove_categories(catids, force=False)**: Removes categories from the dataset.
* catids: list or int: Category IDs to be removed.
* force: boolean. If `force == False`, the category will not be removed if at least one annotation exists with that cat ID.

**reset_catids()**: Resets the category IDs so that they run contiguously from zero. All annotations will be updated to the new numbers.

**remove_videos(videoids, skipverify=False)**: Remove videos from the dataset.
* videoids: list or int: Video IDs to be removed.
* skipverify: boolean. If `True`, the internal consistency verification step will be skipped. This should be used with caution.

**remove_annotations_with_cat(catids)**: Remove all annotations that are in category `catids`.
* catids: list or int.

**verify_data()**: Verify that the data is internally consistent.
* returns True if consistent. Otherwise False.

**write_data(filename, skipverify=False)**: Write the data in ODIMI format. 
* filename: str. Path to output JSON file.
* skipverify: boolean. If `True`, the internal consistency verification step will be skipped. This should be used with caution.

**write_gt_vid(output_folder, videoid=None, filename=None, fps=15)**: Write an mp4 video.
* If no videoID is given, a random video ID will be chosen.
* If no filename is given, it will default to `"ImageNetVID_GT_videoid_%i.mp4" % videoid`

**write_gt_img(imgid, output_dir=None, filename=None, returning=True)**:: Write a single output image.
* returning: boolean. If True, the PIL image will be returned.

**summary()**: Displays a summary of the data.

**set_categories_to_coco()**: Forcibly sets the categories to equal those of the COCO detection challenge.
* This will overwrite the current categories. Use with care.

**translate_cats(catdict)**: Renumbers the categories of the annotations store according to the rules in catdict.
* catdict: dict. keys: current category IDs. Values: New category IDs.
* If a current category is missing from catdict.keys(), the method will not make any changes.

## Example commands
**Create a copy of the DAVIS 2017 training set that only includes categories with annotations**

This continues from the quick start guide example.
```python
import prepare_video_datasets as pvd
od = pvd.OdimiDataset()
# Re-load data from JSON file
od.load_json_data('/data/DAVIS/json/DAVIS-2017-original_train.json')
# Review classes
od.explore_data('cats')
# Remove cats with zero annotations
od.remove_categories([7, 8, 10, 15, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 38, 39, 41, 42, 44, 48, 49, 50, 52, 54, 58, 61, 62, 63, 74, 76, 77])
od.info['version'] = 'DAVIS original training, redundant categories removed.'
od.info['description'] = 'DAVIS train dataset, without any of the categories that are not represented in the training set.'
od.write_data('/data/DAVIS/json/DAVIS-2017-reduced_train.json')
```

**Create a dataset that is DAVIS 2017 train + val, only those classes with annotations**
```python
import dataset_prep.prepare_video_datasets as pvd
davis_tv = pvd.OdimiDataset(version="DAVIS original train+val", description="The complete DAVIS train and validation datasets, combined.")
davis_tv.load_json_data('/data/DAVIS/json/DAVIS-2017-original_train.json')
davis_tv.load_json_data('/data/DAVIS/json/DAVIS-2017-original_val.json')
davis_tv.explore_data('cats')
bad_cats = [7, 15, 17, 19, 22, 25, 26, 27, 28, 30, 32, 34, 38, 39, 41, 42, 44, 49, 50, 52, 58, 61, 62, 63, 74, 76, 77]
davis_tv.remove_categories(bad_cats)
davis_tv.info['version'] = 'DAVIS original train + val, redundant categories removed.'
davis_tv.info['description'] = 'DAVIS train+val dataset, without any of the categories that have no annotations.'
davis_tv.write_data('/data/DAVIS/json/DAVIS-2017-reduced_train+val.json')
```

**Create a dataset for the ImageNet VID 2015 challenge**
```python
import dataset_prep.prepare_video_datasets as pvd
it = pvd.OdimiDataset(version="ImageNet VID original train", description="The complete ImageNet VID train dataset.")
it.load_raw_data('imagenet_vid', 'train', '/data/imagenet/ILSVRC')
it.explore_data('cats')
it.write_data('/data/imagenet/json/imagnet_vid-original_train.json', skipverify=True)
```

**Keep only the first 16 videos**
We assume that you have previously loaded and saved the ImageNet VID validation data.
```python
import random
import dataset_prep.prepare_video_datasets as pvd
iv = pvd.OdimiDataset(version="ImageNet VID original val", description="The complete ImageNet VID train dataset.")
iv.load_json_data('/data/imagenet/json/imagnet_vid-30classes_val.json')
n_keep = 16
remove_vids = random.sample(list(range(554)), 555 - n_keep)
iv.remove_videos(remove_vids)
iv.write_data('/data/imagenet/json/imagnet_vid-30classes_tinyval.json')
```

## ODIMI format
These tools produce data in the ODIMI (Object Detection In Moving Imagery) format, which is an extension to the 
[COCO object detection format](http://cocodataset.org/#format-data).

Top level tags:
```
{
    "info"          : info, 
    "images"        : [image], 
    "annotations"   : [annotation], 
    "licenses"      : [license], 
    "categories"    : [category], 
    "videos"        : [video]
}

```
The *info*, *licenses* and *categories* tags are unchanged from the COCO format.

```
info{
    "year"          : int, 
    "version"       : str, 
    "description"   : str, 
    "contributor"   : str, 
    "url"           : str, 
    "date_created"  : datetime,
}

license{
    "id"            : int, 
    "name"          : str, 
    "url"           : str,
}

category{
    "id"            : int, 
    "name"          : str, 
    "supercategory" : str,
}
```

The new *videos* tags are added to capture the name and IDs of the videos.

```
video{
    "name"          : str,
    "id"            : int,
}

```

*Annotations* and *images* have tags added, to relate the image frame to the video ID.
```
annotation{
    "id"            : int,
    "image_id"      : int,
    "category_id"   : int,
    "segmentation"  : [polygon],
    "area'          : int,
    "iscrowd'       : int,
    "video_id"      : int,
    "bbox"          : [x, y, w, h],
}

image{
    "id"            : int,
    "file_name"     : str,
    "file_path"     : str,
    "video_name"    : str,
    "video_id"      : int,
    "width"         : int,
    "height"        : int,
    "license"       : int,
    "flickr_url"    : str,
    "coco_url"      : str,
    "date_captured" : datetime,
}
```

### Limitations of this code.
The following tags are not populated by this code:
* Annotation: segmentation, area, iscrowd
* Image: width, height, license, flickr_url, coco_url, date_captured.
