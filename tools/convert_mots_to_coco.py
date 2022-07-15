import os
import numpy as np
import json
import cv2


# for MOTS
DATA_PATH = 'datasets/MOTS'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['train', 'test']
import pycocotools.mask as rletools


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test')
        else:
            data_path = os.path.join(DATA_PATH, 'train')
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half

            image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = cv2.imread(os.path.join(data_path, '{}/img1/{:06d}.jpg'.format(seq, i + 1)))
                height, width = img.shape[:2]
                image_info = {'file_name': '{}/img1/{:06d}.jpg'.format(seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            if split != 'test':
                with open(ann_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        fields = line.split(" ")
                        frame_id = int(fields[0])
                        if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                            continue
                        track_id = int(fields[1])
                        cat_id = int(fields[2])
                        if cat_id != 2:
                            continue
                        ann_cnt += 1
                        category_id = 1
                        mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
                        x, y, w, h = rletools.toBbox(mask)
                        ann = {'id': ann_cnt,
                            'category_id': category_id,
                            'image_id': image_cnt + frame_id,
                            'track_id': track_id % 1000,
                            'bbox': [x, y, w, h],
                            'conf': 1.0,
                            'iscrowd': 0,
                            'area': float(w * h)}
                        out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))