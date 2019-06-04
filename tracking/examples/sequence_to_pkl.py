import os
import math
import pickle
import numpy as np

from PIL import Image


def augment_sequence(seq_pkl_path, timestamps_pkl_path, image_dir, images):
    """Adds image_list and pose_list attributes to an RGBDSequence object."""
    with open(seq_pkl_path, 'rb') as seq_file, open(timestamps_pkl_path, 'rb') as time_file:
        seq = pickle.load(seq_file)
        timestamps = pickle.load(time_file)

        image_list = []
        pose_list = []
        for i in range(seq.seq_len):
            filename = get_closest_image(images, timestamps[i])
            img = Image.open(os.path.join(image_dir, filename))
            img.thumbnail((320, 240))
            img_array = np.array(img)
            img_array = np.swapaxes(img_array, 0, 2)
            img_array = np.swapaxes(img_array, 1, 2)
            img_array = np.expand_dims(img_array, axis=0)
            image_list.append(img_array)
            pose_list.append(seq.get_dict(i)['pose'])

        seq.image_list = image_list
        seq.pose_list = pose_list

        return seq


def get_closest_image(images, timestamp):
    """Returns the filename of the image taken closest to the timestamp."""
    min_time = math.inf
    min_image = None
    for img in images:
        diff = abs(timestamp - float(img[:img.rindex('.')]))
        if diff < min_time:
            min_time = diff
            min_image = img
    return min_image


if __name__ == '__main__':
    TRACKING_OUTPUT_PKL = '/home/msamogh/deeptam/deeptam/tracking/examples/sequence.pkl'
    TIMESTAMPS_PKL = '/home/msamogh/deeptam/deeptam/tracking/examples/timestamps.pkl'

    # Load original images
    IMAGES_DIR = '/home/msamogh/deeptam/deeptam/tracking/data/rgbd_dataset_freiburg1_desk/rgb'
    images = sorted(os.listdir(IMAGES_DIR))

    seq = augment_sequence(TRACKING_OUTPUT_PKL, TIMESTAMPS_PKL, IMAGES_DIR, images)
    pickle.dump(seq, open('output.pkl', 'wb'))
