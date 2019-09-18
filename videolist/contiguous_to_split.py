import argparse


def range_to_str(a, b):
    return '%d-%d' % (a, b)


def str_to_range(str):
    return tuple(int(d) for d in str.split('-'))


def main(input_path, output_path, p, m, f):
    input_reader = open(input_path, 'r')
    output_writer = open(output_path, 'w')
    for line in input_reader.readlines():
        line = line.strip()
        video_file_name, video_range = line.split()

        # Note: All ranges are open intervals [start, end), i.e. end is not an indexable frame
        video_range_start, video_range_end = str_to_range(video_range)
        assert(video_range_end-video_range_start == p+m+f)
        preceding_range = (video_range_start, video_range_start + p)
        middle_range = (video_range_start + p, video_range_start + p + m)
        following_range = (video_range_start + p + m, video_range_end)

        output_writer.write('%s %s %s %s\n' % (video_file_name,
                                             range_to_str(*preceding_range),
                                             range_to_str(*middle_range),
                                             range_to_str(*following_range)))

    input_reader.close()
    output_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('p', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('f', type=int)

    args = parser.parse_args()
    main(**vars(args))