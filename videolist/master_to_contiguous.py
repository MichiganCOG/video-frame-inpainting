import argparse


def range_to_str(a, b):
    return '%d-%d' % (a, b)


def str_to_range(str):
    return tuple(int(d) for d in str.split('-'))


def main(input_path, output_path, clip_length, default_stride, first_only):
    input_reader = open(input_path, 'r')
    output_writer = open(output_path, 'w')
    for line in input_reader.readlines():
        line = line.strip()
        video_file_name, video_range = line.split()

        # Note: Video range is a 1-indexed, inclusive range
        video_range_start, video_range_end = str_to_range(video_range)

        # Get the set of possible start indexes, filtering out intervals that fall outside the given range
        # Note: Stride is changed for KTH's running and jogging classes as per Villegas et al. (2017) to keep number of
        # examples per class similar
        stride = 3 if 'running' in video_file_name or 'jogging' in video_file_name else default_stride
        possible_start_indexes = xrange(video_range_start, video_range_end - clip_length + 2, stride)

        for start_index in possible_start_indexes:
            output_writer.write('%s %s\n' % (
                video_file_name,
                range_to_str(start_index, start_index + clip_length - 1)
            ))
            if first_only:
                break

    input_reader.close()
    output_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--clip_length', type=int, default=20)
    parser.add_argument('--default_stride', type=int, default=10)
    parser.add_argument('--first_only', action='store_true')

    args = parser.parse_args()
    main(**vars(args))