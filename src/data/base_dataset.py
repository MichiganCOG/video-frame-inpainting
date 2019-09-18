import os
import random
import re
import traceback
from warnings import warn

import cv2
import imageio
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor

from ..util.util import fore_transform, bgr2gray


class ContiguousVideoClipDataset(data.Dataset):
    def __init__(self, c_dim, video_list_path, seq_length, backwards, flip, image_size, resample_on_fail, padding_size):
        """Constructor

        :param c_dim: The number of color channels each output video should have
        :param video_list_path: The path to the video list text file
        :param K: The number of preceding frames
        :param T: The number of future or middle frames
        :param backwards: Flag to allow data augmentation by randomly reversing videos temporally
        :param flip: Flag to allow data augmentation by randomly flipping videos horizontally
        :param image_size: The spatial resolution of the video (W x H)
        :param F: The number of following frames
        """

        super(ContiguousVideoClipDataset, self).__init__()

        # Initialize basic properties
        self.c_dim = c_dim
        self.backwards = backwards
        self.flip = flip
        self.image_size = image_size
        self.resample_on_fail = resample_on_fail
        self.padding_size = padding_size

        # Read the list of files
        with open(video_list_path, 'r') as f:
            self.files = [line.strip() for line in f.readlines()]

        self.seq_len = seq_length

        self.vid_path = None


    def read_seq(self, vid, frame_indexes, clip_label):
        """Obtain a video clip along with corresponding difference frames and auxiliary information.

        Returns a dict with the following key-value pairs:
        - vid_name: A string identifying the video that the clip was extracted from
        - start-end: The start and end indexes of the video frames that were extracted (inclusive)
        - targets: The full video clip [C x H x W x T FloatTensor]
        - diff_in: The difference frames of the preceding frames in the full video clip [C x H x W x T_P]
        - diff_in_F: The difference frames of the following frames in the full video clip [C x H x W x T_F]

        :param vid: An imageio Reader
        :param stidx: The index of the first video frame to extract clips from
        :param vid_name: A string identifying the video that the clip was extracted from
        :param vid_path: The path to the given video file
        """

        targets = []

        # generate [0, 1] random variable to determine flip and backward
        flip_flag = self.flip and (random.random() > 0.5)
        back_flag = self.backwards and (random.random() > 0.5)

        # read and process in each frame
        for t in frame_indexes:
            # read in one frame from the video
            vid_frame = self.get_frame(vid, t)
            if vid_frame is None:
                warn('Failed to read the given sequence of frames (frame %d in %s)' % (t, vid._filename))
                return None

            # resize frame
            img = cv2.resize(vid_frame, (self.image_size[1], self.image_size[0]))[:, :, ::-1]

            # flip the input frame horizontally
            if flip_flag:
                img = img[:, ::-1, :]

            # pad the image
            img = cv2.copyMakeBorder(img, 0, self.padding_size[0], 0, self.padding_size[1], cv2.BORDER_CONSTANT, -1)

            targets.append(to_tensor(img.copy()))

        # Reverse the temporal ordering of frames
        if back_flag:
            targets = targets[::-1]

        # stack frames and map [0, 1] to [-1, 1]
        target = fore_transform(torch.stack(targets))  # T x C x H x W
        # if number of color channels is 1, use the gray scale image as input
        if self.c_dim == 1:
            target = bgr2gray(target)

        ret = {
            'targets': target,
            'clip_label': clip_label
        }

        return ret


    def open_video(self, vid_path):
        """Obtain a file reader for the video at the given path.

        Wraps the line to obtain the reader in a while loop. This is necessary because it fails randomly even for
        readable videos.

        :param vid_path: The path to the video file
        """
        for _ in xrange(5):
            try:
                vid = imageio.get_reader(vid_path, 'ffmpeg')
                return vid
            except IOError:
                traceback.print_exc()
                warn('imageio failed in loading video %s, retrying' % vid_path)

        warn('Failed to load video %s after multiple attempts, returning' % vid_path)
        return None


    def get_frame(self, vid, frame_index):
        for _ in xrange(5):
            try:
                frame = vid.get_data(frame_index)
                return frame
            except imageio.core.CannotReadFrameError:
                traceback.print_exc()
                warn('Failed to read frame %d in %s, retrying' % (frame_index, vid._filename))

        warn('Failed to read frame %d in %s after five attempts. Check for corruption' % (frame_index, vid._filename))
        return None


    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Obtain data associated with a video clip from the dataset (BGR video frames)."""

        # Try to use the given video to extract clips
        while True:
            # Parse the line for the given index
            split_line = self.files[index].split()
            if len(split_line) == 1:
                video_file_path = split_line[0]
            else:
                video_file_path, full_range_str = split_line

            # Open the video
            vid = self.open_video(video_file_path)
            if vid is None:
                if not self.resample_on_fail:
                    raise RuntimeError('Video at %s could not be opened' % video_file_path)
                # Video could not be opened, so try another video
                index = np.random.randint(0, len(self.files))
                continue

            # Use the whole video or, if specified, only use the provided 1-indexed frame indexes
            # Note: full_range is a 0-indexed, inclusive range
            if len(split_line) == 1:
                full_range = (0, vid.get_length()-1)
            else:
                # Convert to 0-indexed indexes
                full_range = tuple(int(d)-1 for d in full_range_str.split('-'))
            full_range_length = full_range[1] - full_range[0] + 1
            if full_range_length < self.seq_len:
                if not self.resample_on_fail:
                    raise RuntimeError('Interval %s in video %s is too short' % (str(full_range), video_file_path))
                # The video clip length is too short, so try another video
                index = np.random.randint(0, len(self.files))
                continue

            # Randomly choose a sub-range (inclusive) within the full range
            # Note: random.randint(a, b) chooses within closed interval [a, b]
            start_index = random.randint(full_range[0], full_range[1] - self.seq_len + 1)
            frame_indexes = range(start_index, start_index + self.seq_len)
            # Select the chosen frames
            try:
                clip_label = '%s_%d-%d' % (os.path.basename(video_file_path), full_range[0] + 1, full_range[1] + 1)
                item = self.read_seq(vid, frame_indexes, clip_label)
            except IndexError as e:
                warn('IndexError for video at %s, video length %d, video range %s, indexes %s' \
                     % (video_file_path, vid.get_length(), full_range_str, str(frame_indexes)))
                raise e
            if item is None:
                if not self.resample_on_fail:
                    raise RuntimeError('Failed to sample frames starting at %d in %s' % (start_index, video_file_path))
                # Failed to load the given set of frames, so try another item
                index = np.random.randint(0, len(self.files))
                continue

            return item


class DisjointVideoClipDataset(ContiguousVideoClipDataset):
    def __init__(self, c_dim, video_list_path, K, F, image_size, padding_size):
        
        super(DisjointVideoClipDataset, self).__init__(c_dim, video_list_path, None, False, False, image_size, False,
                                                       padding_size)
        
        self.K = K
        self.F = F
    
    def __getitem__(self, index):
        """Obtain data associated with a video clip from the dataset."""

        # Parse the line for the given index
        m = re.match('(.+) (\d+)-(\d+) (\d+)-(\d+)', self.files[index])
        if m is None:
            raise RuntimeError('Expected line from video list to have format "<video_path> <A-B> <C-D>", '
                               'but found line "%s")' % self.files[index])
        video_file_path, p_a, p_b, f_a, f_b = m.group(1, 2, 3, 4, 5)

        # Open the video
        vid = self.open_video(video_file_path)
        if vid is None:
            raise RuntimeError('Video at %s could not be opened' % video_file_path)

        # Convert clip indexes to 0-indexed values
        p_a = int(p_a) - 1
        p_b = int(p_b) - 1
        f_a = int(f_a) - 1
        f_b = int(f_b) - 1

        # Construct the frame indexes to extract
        frame_indexes = range(p_a, p_b + 1) + range(f_a, f_b + 1)
        # Select the chosen frames
        try:
            clip_label = '%s_%d-%d_%d-%d' % (os.path.basename(video_file_path), p_a + 1, p_b + 1, f_a + 1, f_b + 1)
            item = self.read_seq(vid, frame_indexes, clip_label)
        except IndexError as e:
            warn('IndexError for video at %s, video length %d, indexes %s' \
                 % (video_file_path, vid.get_length(), str(frame_indexes)))
            raise e
        if item is None:
            raise RuntimeError('Failed to sample frames %d-%d and %d-%d in %s' % (p_a, p_b, f_a, f_b, video_file_path))

        return item