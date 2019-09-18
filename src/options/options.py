import argparse

import torch


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # general experiment
        exp_arg_group = self.parser.add_argument_group('Experiment parameters')
        exp_arg_group.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment')

        # inputs and outputs
        in_out_arg_group = self.parser.add_argument_group('Model input/output parameters')
        in_out_arg_group.add_argument('--K', type=int, required=True,
                                      help='Length of the preceding sequence (in frames)')
        in_out_arg_group.add_argument('--T', type=int, required=True,
                                      help='Length of the middle sequence (in frames)')
        in_out_arg_group.add_argument('--F', type=int, required=True,
                                      help='Length of the following sequence (in frames)')
        in_out_arg_group.add_argument('--batch_size', type=int, default=4, help='Mini-batch size')
        in_out_arg_group.add_argument('--image_size', type=int,  nargs='+', default=[128],
                                      help='Image size (H x W). Can be specified as two numbers (e.g. "160 208") or '
                                           'one (in which case, H = W)')
        in_out_arg_group.add_argument('--padding_size', type=int, nargs='+', default=[0],
                                      help='Amount of padding to add to the bottom and right sides of the image. Can '
                                           'be specified as two numbers (e.g. "10 32") or one (in which case, '
                                           'pad_bottom = pad_right)')
        in_out_arg_group.add_argument('--c_dim', type=int, default=3, help='Number of channels in the image input')

        # basic dimension
        model_arg_group = self.parser.add_argument_group('Model specification parameters')
        model_arg_group.add_argument('--model_key', type=str, required=True,
                                     help='Key identifying the generator to create')

        # path to save ckpt tb and result
        dir_arg_group = self.parser.add_argument_group('Directory parameters')
        dir_arg_group.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                                   help='Path to store/load checkpoint files')

        # data loading
        common_data_load_arg_group = self.parser.add_argument_group('Common data loading parameters')
        common_data_load_arg_group.add_argument('--num_threads', type=int, default=2,
                                                help='Number of threads used to load data')

    def parse(self, allow_unknown=False):
        if allow_unknown:
          opt, unknown_opt = self.parser.parse_known_args()
          print('Ignored arguments: %s' % str(unknown_opt))
        else:
          opt = self.parser.parse_args()

        if len(opt.image_size) == 1:
            opt.image_size.append(opt.image_size[0])

        if len(opt.padding_size) == 1:
            opt.padding_size.append(opt.padding_size[0])

        # Check that at least one GPU is available
        assert(torch.cuda.is_available())

        return opt


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()

        # optimization parameters
        opt_arg_group = self.parser.add_argument_group('Optimization parameters')
        opt_arg_group.add_argument('--lr', type=float, default=0.0001, help='Base learning rate')
        opt_arg_group.add_argument('--beta1', type=float, default=0.5, help='Momentum term of adam')
        opt_arg_group.add_argument('--max_iter', type=int, default=100000,
                                   help='Maximum number of iterations (batches) to train on')

        # loss params
        loss_arg_group = self.parser.add_argument_group('Loss parameters')
        loss_arg_group.add_argument('--alpha', type=float, default=1.0, help='Image loss weight')
        loss_arg_group.add_argument('--beta', type=float, default=0.02, help='GAN loss weight')

        # freqs
        freq_arg_group = self.parser.add_argument_group('Training frequency parameters')
        freq_arg_group.add_argument('--print_freq', type=int, default=100,
                                    help='Frequency at which to print training results in the console (in iterations)')
        freq_arg_group.add_argument('--save_latest_freq', type=int, default=1000,
                                    help='Frequency at which to save a snapshot of the training environment')
        freq_arg_group.add_argument('--validate_freq', type=int, default=10000,
                                    help='Frequency at which to perform model validation')

        # adversarial training
        adv_train_arg_group = self.parser.add_argument_group('Adversarial training parameters')
        adv_train_arg_group.add_argument('--df_dim', type=int, default=64,
                                         help='Number of filters in first conv layer of the discriminator')
        adv_train_arg_group.add_argument('--Ip', type=int, default=3,
                                         help='Number of power iterations used to compute max singular value in the '
                                              'spectral-normalized discriminator (only used if "--sn" option is used)')
        adv_train_arg_group.add_argument('--disc_window_size', type=int, default=3,
                                         help='The number of video frames that the discriminator sees at a time')

        # training/validation/visualization datasets
        training_data_load_arg_group = self.parser.add_argument_group('Training data loading parameters')
        training_data_load_arg_group.add_argument('--alt_K', type=int, required=True,
                                                  help='An alternative length for the preceding sequence (in frames). '
                                                       'Used to validate and visualize generalization performance')
        training_data_load_arg_group.add_argument('--alt_T', type=int, required=True,
                                                  help='An alternative number of middle frames to predict (in '
                                                       'frames). Used to validate and visualize generalization '
                                                       'performance')
        training_data_load_arg_group.add_argument('--alt_F', type=int, required=True,
                                                  help='An alternative length for the following sequence (in frames). '
                                                       'Used to validate and visualize generalization performance')
        training_data_load_arg_group.add_argument('--train_video_list_path', type=str, required=True,
                                                  help='The path to the text file containing the list of training '
                                                       'video (clips)')
        training_data_load_arg_group.add_argument('--val_video_list_path', type=str, required=True,
                                                  help='The path to the text file containing the list of validation '
                                                       'video (clips)')
        training_data_load_arg_group.add_argument('--val_video_list_alt_T_path', type=str, required=True,
                                                  help='The path to the text file containing the list of validation '
                                                       'video (clips) with an alternative number of middle frames '
                                                       '(given by alt_T)')
        training_data_load_arg_group.add_argument('--val_video_list_alt_K_F_path', type=str, required=True,
                                                  help='The path to the text file containing the list of validation '
                                                       'video (clips) with an alternative number of preceding and '
                                                       'following frames (given by alt_K and alt_F)')
        training_data_load_arg_group.add_argument('--vis_video_list_path', type=str, required=True,
                                                  help='The path to the text file containing the list of visualization '
                                                       'video (clips)')
        training_data_load_arg_group.add_argument('--vis_video_list_alt_T_path', type=str, required=True,
                                                  help='The path to the text file containing the list of '
                                                       'visualization video (clips) with an alternative number of '
                                                       'middle frames (given by alt_T)')
        training_data_load_arg_group.add_argument('--vis_video_list_alt_K_F_path', type=str, required=True,
                                                  help='The path to the text file containing the list of '
                                                       'visualization video (clips) with an alternative number of '
                                                       'preceding and following frames (given by alt_K and alt_F)')
        training_data_load_arg_group.add_argument('--serial_batches', action='store_true',
                                                  help='Flag for loading videos sequentially. If False, videos will be '
                                                       'loaded randomly')
        training_data_load_arg_group.add_argument('--no_backwards', action='store_true',
                                                  help='Flag to turn off data augmentation via playing videos '
                                                       'backwards')
        training_data_load_arg_group.add_argument('--no_flip', action='store_true',
                                                  help='Flag to turn of data augmentation via flipping video frames '
                                                       'horizontally')
        training_data_load_arg_group.add_argument('--sample_KTF', action='store_true',
                                                  help='Flag to sample the number of preceding, middle, and following '
                                                       'frames in each minibatch')

        # Transformer network training parameters
        transformer_network_train_arg_group = self.parser.add_argument_group('Transformer network training parameters')
        transformer_network_train_arg_group.add_argument('--tf_p_min', type=float, default=0,
                                                         help='Minimum teacher-forcing probability')
        transformer_network_train_arg_group.add_argument('--tf_p_max', type=float, default=0,
                                                         help='Maximum teacher-forcing probability')
        transformer_network_train_arg_group.add_argument('--tf_offset', type=float, default=100000,
                                                         help='Amount to shift the teacher-forcing curve')
        transformer_network_train_arg_group.add_argument('--tf_decay', type=float, default=20000,
                                                         help='Amount to slow down the teacher-forcing rate function')

        # SloMo network training parameters
        slomo_network_train_arg_group = self.parser.add_argument_group('SloMo network training parameters')
        slomo_network_train_arg_group.add_argument('--lambda_r', type=float, default=0.8,
                                                         help='Weight for reconstruction loss')
        slomo_network_train_arg_group.add_argument('--lambda_p', type=float, default=0.005,
                                                         help='Weight for perceptual loss')
        slomo_network_train_arg_group.add_argument('--lambda_w', type=float, default=0.4,
                                                         help='Weight for warping loss')
        slomo_network_train_arg_group.add_argument('--lambda_s', type=float, default=1,
                                                         help='Weight for smoothness loss')
        slomo_network_train_arg_group.add_argument('--lr_decay_count', type=int, default=40000,
                                                         help='The number of iterations to perform learning rate decay')
        slomo_network_train_arg_group.add_argument('--lr_decay_rate', type=float, default=0.1,
                                                         help='The decay of learning rate')

        # TensorBoard visualization
        training_vis_arg_group = self.parser.add_argument_group('Training visualization parameters')
        training_vis_arg_group.add_argument('--tensorboard_dir', type=str, default='tb',
                                            help='Path to store TensorBoard log files')


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()

        # test data loading parameters
        test_data_load_arg_group = self.parser.add_argument_group('Test data loading parameters')
        test_data_load_arg_group.add_argument('--test_video_list_path', type=str, required=True,
                                              help='The path to the text file containing the list of test video '
                                                   '(clips)')
        test_data_load_arg_group.add_argument('--disjoint_clips', action='store_true',
                                              help='Flag to indicate that the video list specifies a disjoint clip '
                                                   '(i.e. preceding and following frames only')

        # snapshot
        snapshot_arg_group = self.parser.add_argument_group('Snapshot parameters')
        snapshot_arg_group.add_argument('--snapshot_file_name', type=str, default='model_best.ckpt',
                                        help='The file name of the model to load from')

        # qualitative result destination
        qual_result_arg_group = self.parser.add_argument_group('Qualitative result destination parameters')
        qual_result_arg_group.add_argument('--qual_result_root', type=str, required=True,
                                           help='The root path where qualitative results will be stored')

        # output
        output_arg_group = self.parser.add_argument_group('Output parameters')
        output_arg_group.add_argument('--intermediate_preds', action='store_true',
                                      help='Flag to write intermediate predictions in addition to final ones')