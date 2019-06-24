import os
import json
import datetime as dt
from read_dataset_davis import read_davis
from read_dataset_imagenet_vid import read_imagenet
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    print("You need cv2 in order to create videos. Install using 'pip install opencv-python'")

import warnings
warnings.filterwarnings("error")


class OdimiDataset:
    """
    Class for converting video datasets into a format usable by ODIMI.
    Based on the COCO annotaitons (http://cocodataset.org/#format-data), modified for videos.
    """

    def __init__(self, version="", description=""):
        """
        Initialise class.
        """
        self.images = []
        self.annotations = []
        self.videos = []
        self.categories = []
        self.info = {"year": 2019, "version": version, "description": description, "contributor": "", "url": "",
                     "date_created": dt.datetime.now().isoformat(), }
        licenses = [{"id": 1, "name": "", "url": "", }]
        self.export_data = {"licenses": licenses}

        self.config = {'allowed_styles': ['davis', 'imagenet_vid', 'cityscapes', 'gygo']}
        self.vid2imgs = {}
        self.img2anns = {}

    def __str__(self):
        """
        String
        :return:
        """
        return ("ODIMI format data. \nVersion: %s \nDescription: "
                "%s \nThere are %i loaded images." % (self.info['version'], self.info["description"], len(self.images)))

    def __len__(self):
        """
        Number of images
        :return:
        """
        return len(self.images)

    def _summarise_data(self):
        print(self.__str__())
        print("Summary of data:")
        print("  Categories:  %7i \n  Videos:      %7i \n"
              "  Images:      %7i \n  Annotations: %7i" % (len(self.categories), len(self.videos),
                                                           len(self.images), len(self.annotations)))

    def load_raw_data(self, data_style, data_part, data_dir):
        """
        Load data into memory
        :param data_part:
        :param data_style: davis | imagenet_vid | cityscapes | gygo
        :param data_dir: path
        :return:
        This function populates self.images, .annotations, .videos, .categories
        """
        data_style = data_style.lower()

        assert data_style in self.config['allowed_styles'], "data_style must be one of: %s" % ", ".join(
            self.config['allowed_styles'])

        # package existing lengths
        existing_lens = {'cats': len(self.categories), 'vids': len(self.videos),
                         'imgs': len(self.images), 'anns': len(self.annotations)}

        if data_style == 'davis':
            print(" Reading raw data from %s..." % data_dir)
            cats, vids, imgs, anns = read_davis(data_dir, data_part, existing_lens, year=2017, data_res='full')
        elif data_style == 'imagenet_vid':
            print(" Reading raw data from %s..." % data_dir)
            cats, vids, imgs, anns = read_imagenet(data_dir, data_part, existing_lens)
        else:
            raise NotImplementedError("data_style %s has not been implemented." % data_style)

        # Merge cats, vids, imgs, anns with any values that might already exist in the cats.
        # Otherwise just extend the lists
        print(" %i videos, %i frames and %i cats have been read." % (len(vids), len(imgs), len(cats)))
        print(" Merging data with existing data store.")
        self._merge_data(cats, vids, imgs, anns)
        self._summarise_data()

    def load_json_data(self, filename):
        """
        Load previously prepared data.
        :param filename:
        :return:
        """
        with open(filename) as f:
            data = json.load(f)

        self._merge_data(data['categories'], data['videos'], data['images'], data['annotations'])
        self.info = data['info']
        licenses = data['licenses']
        self.export_data = {"licenses": licenses}

        self._summarise_data()

    def explore_data(self, aspect, catslist=None):
        """
        Prints an aspect of the data
            all: A top level summary
            cats: A list of the categories and IDs, and the number of occurrences
            videos: A list of videos, and the categories they contain
            vids_with_cats: A list of videos, that contain any of the cats in catslist
        :param aspect: all | cats | videos
        :return:
        """
        aspect = aspect.lower()
        if aspect == 'all':
            self._summarise_data()
        elif aspect == 'cats' or aspect == 'categories':
            self._print_cats_occurances()
        elif aspect == 'vids_with_cats':
            assert catslist is not None, "a catslist must be supplied with aspect='vids_with-cats'"
            vidids = self._get_vids_with_catids(catslist)
            vidid2name = self._get_vidid_name_dict()
            print("The following videos contain at least one of the classes in catslist")
            print("video ID; video name")
            for id_ in vidids:
                print("%i; %s" % (id_, vidid2name[id_]))
        elif aspect == 'vids_only_with_cats':
            assert catslist is not None, "a catslist must be supplied with aspect='vids_with-cats'"
            vidids = self._get_vids_onlywith_catids(catslist)
            vidid2name = self._get_vidid_name_dict()
            print("The following videos contain only one (or more) of the classes in catslist")
            print("video ID; video name")
            vidsids_str = [str(id_) for id_ in vidids]
            for id_ in vidids:
                print("%i; %s" % (id_, vidid2name[id_]))
            str_out = ", ".join(vidsids_str)
            print(str_out)
        else:
            print("Explore data aspect %s unknown." % aspect)

    def remove_categories(self, catids, force=False):
        """
        Remove a category.
        :param catids: list (or int) of IDs to remove
        :param force: if False, will not remove unless there are no instances.
        :return:
        """
        if type(catids) == int:
            catids = [catids]
        for catid in catids:
            count = self._count_anns_with_catid(catid)
            if count == 0 or force:
                self._remove_category(catid)
            else:
                print("Cannot remove category ID %i as %i annotations exist with that cat. "
                      "Use force=True to force." % (catid, count))
        self.reset_catids()

    def reset_catids(self):
        """
        Reset the category IDs so that they run continuously from zero. Update references in images.
        :return:
        """
        old_cats = self.categories
        old_anns = self.annotations
        new_cats = []
        new_anns = []

        cat_id_change = {}  # To store old_cat :new_cat. This will be applied to annotations in anns.
        current_cat_ids = [cc['id'] for cc in old_cats]

        # Work out which of the current cats have correct ids, and which need to be changed.
        for newid, catid in enumerate(current_cat_ids):
            if newid != catid:
                # add mapping to cat_id_change
                cat_id_change[catid] = newid

        # Do any of our cat_ids need changing? If not, do nothing.
        if len(cat_id_change) == 0:
            return
        print(" %i categories need to be relabelled." % len(cat_id_change))

        # Work through the cats list, changing the ID as required by cat_id_change
        ids_to_change = list(cat_id_change.keys())
        for cat in old_cats:
            this_id = cat['id']
            if this_id in ids_to_change:
                # Change this ID
                cat['id'] = cat_id_change[this_id]
                new_cats.append(cat)
            else:
                new_cats.append(cat)

        # Work through anns list, checking if the cat needs to be changed.
        for ann in old_anns:
            current_cat = ann['category_id']
            if current_cat in ids_to_change:
                # We have to change this cat id
                ann['category_id'] = cat_id_change[current_cat]
                new_anns.append(ann)
            else:
                # we do not have to change anything.
                new_anns.append(ann)

        # Update internal store.
        self.categories = new_cats
        self.annotations = new_anns

        self._summarise_data()

    def remove_videos(self, videoids, skipverify=False):
        """
        Removes a video by IDs or name. This will remove the video entry, images from that video, and annotations.
        It will not re-number the images or videos.
        :param videoids: valid video ID
        :return:
        """
        if type(videoids) == int:
            videoids = [videoids]

        # remove annotations
        new_anns = []
        removed_anns_count = 0
        for ann in self.annotations:
            if ann['video_id'] in videoids:
                removed_anns_count += 1
            else:
                new_anns.append(ann)
        self.annotations = new_anns

        # remove images
        new_imgs = []
        removed_imgs_count = 0
        for img in self.images:
            if img['video_id'] in videoids:
                removed_imgs_count += 1
            else:
                new_imgs.append(img)
        self.images = new_imgs

        # Remove videos
        new_vids = []
        removed_vids_count = 0
        for vid in self.videos:
            if vid['id'] in videoids:
                removed_vids_count += 1
            else:
                new_vids.append(vid)
        self.videos = new_vids

        # Report numbers
        print(" %7i annotations have been removed; %i remain." % (removed_anns_count, len(self.annotations)))
        print(" %7i images have been removed; %i remain." % (removed_imgs_count, len(self.images)))
        print(" %7i videos have been removed; %i remain." % (removed_vids_count, len(self.videos)))
        if not skipverify:
            verify = self.verify_data()
            if verify:
                "Data verification passed."
            else:
                "WARNING: Data verification failed."
        self._summarise_data()

    def remove_annotations_with_cat(self, catids):
        """
        Remove all annotations with a cat inside catids.
        :param catids:
        :return:
        """
        if type(catids) == int:
            catids = [catids]
        # remove annotations
        new_anns = []
        removed_anns_count = 0
        for ann in self.annotations:
            if ann['category_id'] in catids:
                removed_anns_count += 1
            else:
                new_anns.append(ann)
        self.annotations = new_anns
        self._summarise_data()

    def verify_data(self):
        """
        Verify that the data is self-consistent.
        :return: True if data correct.
        """
        # Ensure that cat IDs start from zero, and then proceed contiguously to the max.
        cat_ids_found = []
        for cat in self.categories:
            cat_ids_found.append(cat['id'])
        # Order this list
        cats_found = list(set(cat_ids_found))
        # Check that the numbers are contiguous
        for expected, cat in enumerate(cats_found):
            if expected != cat:
                print("Category with ID %i should be %i." % (cat, expected))
                print("cats_found list:", cats_found)
                return False

        # For each annotation, ensure that the cat exists, and so does the relevent image
        imgs_ids_found = []
        for ann in self.annotations:
            this_cat = ann['category_id']
            if this_cat not in cats_found:
                print("ann category not in found list")
                print("This ann:", ann)
                print("Cats found:", cats_found)
                return False
            imgs_ids_found.append(ann['image_id'])
        imgs_ids_found = list(set(imgs_ids_found))

        print("   Images with at least one annotation: %i" % len(imgs_ids_found))

        # For each of imgs_ids_found, mmake sure they exist
        images_ids_in_images={}
        for img in self.images:
            img_id = img['id']
            images_ids_in_images[img_id] = True

        for ii, imgid in enumerate(imgs_ids_found):
            try:
                images_ids_in_images[imgid]
            except KeyError:
                print("annotations contains a reference to imgid %i, but this was not found in self.images" % imgid)
                return False
            if ii % 250000 == 0 and ii > 0:
                print("   Examined %i of %i frames." % (ii, len(self.images)))
        print("   Examined %i of %i frames." % (ii, len(self.images)))

        print(" \nData validation passed!")
        return True

    def write_data(self, filename, skipverify=False):
        """
        Write the JSON data to filename
        :param filename: filename. Location must exist.
        :return:
        """
        if not skipverify:
            verify = self.verify_data()
            if verify:
                print("Data verification passed.")
            else:
                assert verify, "Data verification has failed."

        # Update export_data
        self.export_data['info'] = self.info
        self.export_data['images'] = self.images
        self.export_data['annotations'] = self.annotations
        self.export_data['categories'] = self.categories
        self.export_data['videos'] = self.videos

        with open(filename, 'w') as outfile:
            json.dump(self.export_data, outfile)

    def write_gt_vid(self, output_folder, videoid=None, filename=None, fps=15):
        """
        Write a video of videoname to filename.
        :param output_folder: Folder to save video.
        :param videoid: ID of video. If blank, random ID is selected.
        :param filename: Overwrite default filename
        :param fps: FPS to save video.
        :return:
        """
        assert os.path.isdir(output_folder), "Output folder %s is not accessible." % output_folder
        video_ids = self._get_vidids_list()
        if videoid:
            assert videoid in video_ids
        else:
            videoid = random.choice(video_ids)

        if not filename:
            filename = "ImageNetVID_GT_videoid_%i.mp4" % videoid
        filepath = os.path.join(output_folder, filename)

        # Create vid2imgs:
        if not self.vid2imgs:
            self._build_vid2imgs_dict()
        # Create img2anns:
        if not self.img2anns:
            self._build_img2anns_dict()

        # Get list of frames and anns
        our_frames = self.vid2imgs[videoid]
        our_frames.sort()

        frames = []
        # Create frames in frame dir
        for frame in our_frames:
            frame_data = self.write_gt_img(frame, returning=True)
            frames.append(np.array(frame_data))

        # Compile frames into video.
        first_height, first_width, first_channels = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(filepath, fourcc, fps, (first_width, first_height))
        for frame_data in frames:
            out.write(cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))

        out.release()
        print("  file (%i frames at %i x %i x %i) written to %s" % (
        len(frames), first_width, first_height, first_channels, filepath))


    def write_gt_img(self, imgid, output_dir=None, filename=None, returning=True):
        """
        Write GT image to output_dir
        :param imgid: image id
        :param output_dir:
        :param filename: If not provided, a filename will be created.
        :return: filepath to written file.
        """

        # Get img dict
        im = self._get_image_from_imageid(imgid)
        # get list of annotations for this image ID.
        if not self.img2anns:
            self._build_img2anns_dict()
        anns = self.img2anns[imgid]

        filepath = im['file_path']

        image = Image.open(filepath).convert("RGB")
        # image = Image.open(filepath)
        draw = ImageDraw.Draw(image)

        for ann in anns:
            [xmin, ymin, w, h] = ann['bbox']
            xmax = int(round(xmin + w))
            ymax = int(round(ymin + h))
            catid = ann['category_id']

            draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)],
                      fill=(53, 111, 19), width=2)
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
            draw.text((xmin + 3, ymin - 6), self._get_cat_name(catid), (255, 255, 255), font=font)

        if output_dir and filename:
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)

        if returning:
            return image

        print("WARNING: write_gt_img must be given either a output_dir and filename, or returning=True")

    def summary(self):
        self._summarise_data()

    def set_categories_to_coco(self):
        """Overwrite self.categories with COCO."""
        with open('dataset_prep/coco_classes.txt', 'r') as fid:
            lines = fid.readlines()
        new_cats = []
        for ii, line in enumerate(lines):
            new_cats.append({'name': line.strip(), 'id': ii, 'supercategory': 'object'})
        self.categories = new_cats

    def translate_cats(self, catdict):
        """
        Translate categories in self.annotations into catdict[old].
        :param catdict: dict relating old cat to new.
            If an old cat is not in catdict, self.categories will not be changed.
        :return:
        """
        print("Replacing the categories that annotations point to, using catdict.")
        new_anns = []
        for ann in self.annotations:
            try:
                ann['category_id'] = catdict[ann['category_id']]
            except KeyError:
                print(" Key %i is not in catdict." % ann['category_id'])
                print(" self.annotations HAS NOT been changed.")
                return
            new_anns.append(ann)
        self.annotations = new_anns



    def _merge_data(self, cats, vids, imgs, anns):
        """
        cats, vids, imgs, anns contain newly read data.
        This function integrates them into the existing stores, ensuring that categories aren't duplicated.
        :param cats:
        :param vids:
        :param imgs:
        :param anns:
        :return:
        """
        # Create new lists, which will be used to extend the existing strores.
        new_cats = []
        cat_id_change = {}  # To store old_cat :new_cat. This will be applied to annotations in anns.
        current_cats = self._get_cats_list()

        # Work through cats, checking whether it exists in the current categories.
        for cat in cats:
            if cat['name'] in current_cats:
                # We have a clash, replace catID with the current cat ID.
                old_id = cat['id']
                new_id = self._get_cat_id(cat['name'])
                # Add this change to cat_id_change
                cat_id_change[old_id] = new_id
                # We do not append this to new_cats, as we do not wish to include it.
            else:
                # No clash, append the current cat to new_cats.
                new_cats.append(cat)

        # These will only be changed IF new_cats has changed.
        new_anns = []
        if len(cat_id_change) == 0:
            new_anns = anns
        else:
            # We have to update the annotations.
            cats_to_change = list(cat_id_change.keys())
            for ann in anns:
                current_cat = ann['category_id']
                if current_cat in cats_to_change:
                    # We have to change this cat id
                    ann['category_id'] = cat_id_change[current_cat]
                    new_anns.append(ann)
                else:
                    # we do not have to change anything.
                    new_anns.append(ann)

        print("  %i new categories have been added to self.categories." % len(new_cats))
        print("  %i categories already found in self.categories." % (len(cats) - len(new_cats)))

        self.categories.extend(new_cats)
        self.videos.extend(vids)
        self.images.extend(imgs)
        self.annotations.extend(new_anns)

    def _build_vid2imgs_dict(self):
        """Create dict relating vidID to img IDs"""
        vidids = self._get_vidids_list()
        for vidid in vidids:
            self.vid2imgs[vidid] = []
        for img in self.images:
            # Append to the correct list.
            self.vid2imgs[img['video_id']].append(img['id'])

    def _build_img2anns_dict(self):
        imgids = self._get_imgids_list()
        for imgid in imgids:
            self.img2anns[imgid] = []
        for ann in self.annotations:
            self.img2anns[ann['image_id']].append(ann)


    def _get_cats_list(self):
        """Return list of cats"""
        return [cc['name'] for cc in self.categories]

    def _get_vids_list(self):
        """Return list of video names"""
        return [cc['name'] for cc in self.videos]

    def _get_vidids_list(self):
        """Return list of video ids"""
        return [cc['id'] for cc in self.videos]

    def _get_imgids_list(self):
        """Return list of img ids"""
        return [img['id'] for img in self.images]

    def _get_cat_id(self, cat_name):
        """Return ID of cat"""
        for cc in self.categories:
            if cc['name'] == cat_name:
                return cc['id']
        raise SyntaxError("cat_name %s not found in self.categories." % cat_name)

    def _get_cat_name(self, catid):
        """Return name of cat"""
        for cc in self.categories:
            if cc['id'] == catid:
                return cc['name']
        raise SyntaxError("Cat ID %i not found in self.categories." % catid)

    def _get_vidid_name_dict(self):
        """Return a dictionary relating video ID to name"""
        vidid2name = {}
        for vid in self.videos:
            vidid2name[vid['id']] = vid['name']
        return vidid2name

    def _get_vids_with_catids(self, catids):
        """Return the id of videos with catids"""
        if type(catids) == int:
            catids = [catids]
        found_vid_ids = []
        for ann in self.annotations:
            if ann['category_id'] in catids:
                found_vid_ids.append(ann['video_id'])
        return list(set(found_vid_ids))

    def _get_vids_onlywith_catids(self, catids):
        """Return the id of videos which ONLY contain catids"""
        if type(catids) == int:
            catids = [catids]
        found_vid_ids = self._get_vids_with_catids(catids)
        # Now check which of these vid IDs ONLY contain catids
        confirmed_vid_ids = []
        for vidid in found_vid_ids:
            ann_found_outside_catids = False # If this remains False, do not add to confirmed_vid_ids
            for ann in self.annotations:
                if ann['category_id'] not in catids and ann['video_id'] == vidid:
                    ann_found_outside_catids = True
            if not ann_found_outside_catids:
                confirmed_vid_ids.append(vidid)
        return confirmed_vid_ids

    def _get_image_from_imageid(self, imageid):
        for img in self.images:
            if img['id'] == imageid:
                return img
        print("WARNING: problem with imageid:", imageid)
        raise IOError("Image ID %i not found in self.images" % imageid)


    def _print_cats_occurances(self):
        """cats: A list of the categories and IDs, and the number of occurrences"""
        cat_incidence_count = {}
        for cat in self.categories:
            cat_incidence_count[cat['id']] = 0

        for ann in self.annotations:
            this_cat = ann['category_id']
            cat_incidence_count[this_cat] += 1

        # Now print out answeres
        print("catID; cat name; supercat; annotations count")
        for cat in self.categories:
            print("%i; %s; %s; %i" % (cat['id'], cat['name'], cat['supercategory'], cat_incidence_count[cat['id']]))

        cats_no_incidence = []
        for cat in self.categories:
            if cat_incidence_count[cat['id']] == 0:
                cats_no_incidence.append(str(cat['id']))
        print("All categories with no incidence:")
        print(", ".join(cats_no_incidence))

    def _count_anns_with_catid(self, catid):
        counter = 0
        for ann in self.annotations:
            this_cat = ann['category_id']
            if this_cat == catid:
                counter += 1
        return counter

    def _remove_category(self, catid):
        old_cats = self.categories
        new_cats = []
        for cat in old_cats:
            if cat['id'] != catid:
                new_cats.append(cat)
        self.categories = new_cats