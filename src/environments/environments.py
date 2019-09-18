import os
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from torch.autograd import Variable

from ..discriminators.SNDiscriminator import SNDiscriminator
from ..losses.losses import GDL
from ..models.self_attention.self_attention import BaseSCTSkipConFillInModel
from ..models.tai.tai import TAIFillInModel
from ..models.mcnet.mcnet import MCNetFillInModel
from ..models.optical_flow_fill_in.OFFillInModel import OFFillInModel
from ..util.util import inverse_transform, move_to_devices, weights_init, merge_dicts, as_variable
from ..models.slomo.slomo import SloMoFillInModel
from ..models.slomo.slomo import FlowWarper
from ..models.twi.twi import TimeWeightedInterpolationFillInModel
from ..models.bi_sa.bi_sa import BidirectionalSimpleAverageFillInModel
from ..models.bi_twa.bi_twa import BidirectionalTimeWeightedAverageFillInModel
from ..models.tw_p_f.tw_p_f import TimeWeightedPFFillInModel


def create_training_environment(fill_in_model, c_dim, checkpoints_dir, name, max_K, max_T, max_F, image_size, alpha,
                                beta, lr, beta1, df_dim, Ip, disc_window_size, tf_p_min, tf_p_max, tf_offset,
                                tf_decay, padding_size, lambda_r, lambda_p, lambda_w, lambda_s, lr_decay_count, lr_decay_rate):

    if isinstance(fill_in_model, (TAIFillInModel, TimeWeightedInterpolationFillInModel, BidirectionalSimpleAverageFillInModel, BidirectionalTimeWeightedAverageFillInModel)):
        env = TAITrainingEnvironment(fill_in_model, checkpoints_dir, name, image_size, c_dim, alpha, beta, lr, beta1,
                                     df_dim, Ip, disc_window_size, max_K, max_T, max_F, padding_size)
    elif isinstance(fill_in_model, BaseSCTSkipConFillInModel):
        env = SequentialConvTransformerTrainingEnvironment(fill_in_model, checkpoints_dir, name, image_size, c_dim,
                                                           alpha, beta, lr, beta1, Ip, disc_window_size, 1,
                                                           3, 256, 2048, max_K, max_T, max_F, tf_p_min, tf_p_max,
                                                           tf_offset, tf_decay, padding_size)
    elif isinstance(fill_in_model, MCNetFillInModel):
        env = MCNetTrainingEnvironment(fill_in_model, checkpoints_dir, name, image_size, c_dim, alpha, beta, lr,
                                       beta1, df_dim, Ip, disc_window_size, max_K, max_T, max_F, padding_size)
    elif isinstance(fill_in_model, SloMoFillInModel):
        env = SloMoTrainingEnvironment(fill_in_model, checkpoints_dir, name, lr, beta1, max_K, max_T, max_F,
                                       padding_size, lambda_r, lambda_p, lambda_w, lambda_s, lr_decay_count, lr_decay_rate)
    else:
        raise RuntimeError('Tried to create a training environment for object of unsupported type %s' %
                           type(fill_in_model).__name__)

    # Load latest snapshot if available
    if os.path.isfile(os.path.join(checkpoints_dir, name, 'model_latest.ckpt')):
        print('Loading latest snapshot...')
        env.load('model_latest.ckpt')
    print('Loaded training environment')

    return env


def create_eval_environment(fill_in_model, checkpoints_dir, name, snapshot_file_name, padding_size):
    env = BaseVideoFillInEnvironment(fill_in_model, checkpoints_dir, name, padding_size)
    if not isinstance(fill_in_model, (OFFillInModel, TimeWeightedPFFillInModel)):
        env.load(snapshot_file_name)
    print('Loaded evaluation environment')

    return env


class BaseVideoFillInEnvironment(object):
    def __init__(self, video_fill_in_model, checkpoints_dir, name, padding_size):
        """Constructor

        :param video_fill_in_model: The fill-in model. Must be a nn.Module whose forward() method takes in T,
                                    preceding_frames, and following_frames
        :param checkpoints_dir: The root folder where checkpoints are stored
        :param name: The name of the experiment
        """

        self.save_dir = os.path.join(checkpoints_dir, name)
        self.padding_size = padding_size

        # define generator
        self.generator = video_fill_in_model
        self.generator = move_to_devices(self.generator)
        self.generator.apply(weights_init)

        self.K = None
        self.T = None
        self.F = None

    def forward_test(self):
        """Forward the current inputs through the video frame inpainting or prediction model."""

        self.gen_output = self.generator(self.T, self.preceding_frames, self.following_frames)

    def set_test_inputs(self, preceding_frames, following_frames):
        """Set the current data to use for computing fake videos, losses, etc."""

        self.preceding_frames = Variable(preceding_frames.contiguous().cuda(async=True), volatile=True)
        self.following_frames = Variable(following_frames.contiguous().cuda(async=True), volatile=True)

    def set_gt_middle_frames_test(self, gt_middle_frames):
        self.gt_middle_frames = Variable(gt_middle_frames.contiguous().cuda(async=True))

    def load(self, snapshot_file_name):
        """Load a snapshot of the environment.

        :param snapshot_file_name: The name of the snapshot to load
        """

        save_path = os.path.join(self.save_dir, snapshot_file_name)
        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            snapshot = torch.load(save_path)
        else:
            raise RuntimeError('Failed to find snapshot at path %s' % save_path)

        self.generator.load_state_dict(snapshot['generator'])

        return snapshot

    def eval(self):
        """Sets the generator model to evaluation mode (e.g. affects dropout and batch-norm layers)."""
        self.generator.eval()


class BaseTrainingEnvironment(BaseVideoFillInEnvironment):
    def __init__(self, fill_in_model, checkpoints_dir, name, lr, beta1, max_K, max_T, max_F, padding_size):
        """Constructor

        :param fill_in_model: A tuple of arguments used to initialize the generator associated with this environment
        :param checkpoints_dir: The root folder where checkpoints are stored
        :param name: The name of the experiment
        :param image_size: The spatial resolution of the video
        :param c_dim: The number of color channels (e.g. 3 for RGB)
        :param alpha: The weight of the image reconstruction-based loss
        :param beta: The weight of the adversarial/discriminator-based loss
        :param lr: The learning rate of the optimizers
        :param beta1: The first beta term used by the Adam optimizer
        :param df_dim: Controls the number of features in each layer of the discriminator
        :param Ip: The number of power iterations to use when computing max singular value (used if sn is True)
        :param disc_t: The total number of frames per video that the discriminator will take in
        """

        super(BaseTrainingEnvironment, self).__init__(fill_in_model, checkpoints_dir, name, padding_size)

        # training setting
        self.start_update = 0
        self.total_updates = 0
        self.start_sum_avg_psnr_err = 0
        self.start_sum_avg_ssim_err = 0

        self.max_K = max_K
        self.max_T = max_T
        self.max_F = max_F

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))


    def sample_KTF(self, allow_random_sampling):
        if allow_random_sampling:
            K = np.random.randint(1, self.max_K+1)
            T = np.random.randint(1, self.max_T+1)
            F = np.random.randint(1, self.max_F+1)
        else:
            K = self.max_K
            T = self.max_T
            F = self.max_F

        return K, T, F

    def set_train_inputs(self, preceding_frames, following_frames, gt_middle_frames):
        self.preceding_frames = Variable(preceding_frames.contiguous().cuda(async=True))
        self.following_frames = Variable(following_frames.contiguous().cuda(async=True))
        self.gt_middle_frames = Variable(gt_middle_frames.contiguous().cuda(async=True))

    def forward_train(self):
        """Forward the current inputs through the video frame inpainting or prediction model."""

        self.gen_output = self.generator(self.T, self.preceding_frames, self.following_frames)

    def get_current_state_dict(self, total_updates, sum_avg_psnr_err, sum_avg_ssim_err):
        """Get a dict defining the current state of training (used for snapshotting).

        :param total_updates: The number of training iterations performed so far
        :param sum_avg_psnr_err: Take the average PSNR across all videos, then sum across time
        :param sum_avg_ssim_err: Take the average SSIM across all videos, then sum across time
        """

        current_state = {
            'updates': total_updates,
            'sum_avg_psnr_err': sum_avg_psnr_err,
            'sum_avg_ssim_err': sum_avg_ssim_err,
            'generator': self.generator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict()
        }

        return current_state

    def load(self, snapshot_file_name):
        """Load a snapshot of the environment.

        :param snapshot_file_name: The name of the snapshot to load
        """
        snapshot = super(BaseTrainingEnvironment, self).load(snapshot_file_name)
        self.start_update = snapshot['updates']
        self.start_sum_avg_psnr_err = snapshot['sum_avg_psnr_err']
        self.start_sum_avg_ssim_err = snapshot['sum_avg_ssim_err']
        self.optimizer_G.load_state_dict(snapshot['optimizer_G'])

        return snapshot

    def save(self, snapshot_file_name, total_updates, sum_avg_psnr_err, sum_avg_ssim_err):
        """Save the current state of the environment.

        :param snapshot_file_name: A name for the snapshot to save
        :param total_updates: The number of training iterations performed so far
        :param sum_avg_psnr_err: Take the average PSNR across all videos, then sum across time
        :param sum_avg_ssim_err: Take the average SSIM across all videos, then sum across time
        """
        current_state = self.get_current_state_dict(total_updates, sum_avg_psnr_err, sum_avg_ssim_err)
        save_path = os.path.join(self.save_dir, snapshot_file_name)
        torch.save(current_state, save_path)


    def optimize_parameters(self):
        """Perform one generator update step."""

        self.optimizer_G.zero_grad()
        self.compute_loss_G()
        self.loss_G.backward()
        self.optimizer_G.step()

    def compute_loss_G(self):
        """Compute the generator's loss."""
        self.loss_G = as_variable(torch.zeros(1)).cuda()

    def get_current_errors(self):
        """Obtain a list that specifies the losses associated with training."""

        error_dict = dict([
            ('G_loss', self.loss_G.data[0]),
        ])

        return error_dict

    def get_current_visuals(self):
        """Obtain a dict of video tensors to visualize in TensorBoard."""

        pred_t = self.gen_output['pred']
        pred_vis_seq = torch.cat([self.preceding_frames, pred_t, self.following_frames], dim=1)
        target_vis_seq = torch.cat([self.preceding_frames, self.gt_middle_frames, self.following_frames], dim=1)

        vis_dict = OrderedDict([
            ('pred_vis_seq', pred_vis_seq),
            ('target_vis_seq', target_vis_seq)
        ])

        return vis_dict

    def train(self):
        """Sets the generator model to training mode (e.g. affects dropout and batch-norm layers)."""
        self.generator.train()


class L2GDLDiscTrainingEnvironment(BaseTrainingEnvironment):
    """Base training environment when L2, image gradient difference (GDL), and discriminator losses are used."""

    def __init__(self, fill_in_model, checkpoints_dir, name, image_size, c_dim, alpha, beta, lr, beta1, df_dim, Ip,
                 disc_t, max_K, max_T, max_F, padding_size):
        super(L2GDLDiscTrainingEnvironment, self).__init__(fill_in_model, checkpoints_dir, name, lr, beta1, max_K,
                                                           max_T, max_F, padding_size)

        self.loss_Lp = torch.nn.MSELoss()
        self.loss_gdl = GDL()
        self.loss_d = torch.nn.BCEWithLogitsLoss()

        self.alpha = alpha
        self.beta = beta

        self.disc_t = disc_t

        # define discriminator
        # Use spectral-normalized discriminator
        discriminator = SNDiscriminator((image_size[0] + padding_size[0], image_size[1] + padding_size[1]), c_dim,
                                        disc_t, df_dim, Ip)
        discriminator = move_to_devices(discriminator)
        discriminator.apply(weights_init)
        self.discriminator = discriminator

        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))


    def get_current_state_dict(self, total_updates, sum_avg_psnr_err, sum_avg_ssim_err):
        current_state = super(L2GDLDiscTrainingEnvironment, self).get_current_state_dict(total_updates,
                                                                                         sum_avg_psnr_err,
                                                                                         sum_avg_ssim_err)
        current_state['discriminator'] = self.discriminator.state_dict()
        current_state['optimizer_D'] = self.optimizer_D.state_dict()

        return current_state


    def load(self, snapshot_file_name):
        snapshot = super(L2GDLDiscTrainingEnvironment, self).load(snapshot_file_name)
        self.discriminator.load_state_dict(snapshot['discriminator'])
        self.optimizer_D.load_state_dict(snapshot['optimizer_D'])

        return snapshot


    def create_fake_labels(self):
        """Construct the ground-truth labels corresponding to sliding the discriminator across the current batch.

        :return: B x (K+T+F-disc_t+1) FloatTensor
        """

        num_ones_P = max(0, self.K - self.disc_t + 1)
        num_ones_F = max(0, self.F - self.disc_t + 1)
        fake_labels_list = []
        if num_ones_P > 0:
            fake_labels_list.append(torch.ones(num_ones_P))
        fake_labels_list.append(torch.zeros(self.K + self.T + self.F - self.disc_t + 1 - num_ones_P - num_ones_F))
        if num_ones_F > 0:
            fake_labels_list.append(torch.ones(num_ones_F))

        return torch.cat(fake_labels_list)


    def compute_loss_D(self):
        """Compute the discriminator's loss on real and fake videos, and backprop the loss through the discriminator."""

        # fake
        input_fake = torch.cat([self.preceding_frames, self.gen_output['pred'], self.following_frames], dim=1)
        input_fake_ = input_fake.detach()
        h = self.discriminator(input_fake_)
        B = input_fake.size(0)
        fake_labels = self.create_fake_labels()
        fake_labels = as_variable(fake_labels.view(1, fake_labels.size(0)).expand(B, fake_labels.size(0))).cuda()
        self.loss_d_fake = self.loss_d(h, fake_labels)

        # real
        input_real = torch.cat([self.preceding_frames, self.gt_middle_frames, self.following_frames], dim=1)
        input_real_ = input_real.detach()
        h_ = self.discriminator(input_real_)
        labels_ = Variable(torch.ones(h_.size())).cuda()
        self.loss_d_real = self.loss_d(h_, labels_)

        self.loss_D = self.loss_d_fake + self.loss_d_real


    def optimize_parameters(self):

        super(L2GDLDiscTrainingEnvironment, self).optimize_parameters()

        self.optimizer_D.zero_grad()
        self.compute_loss_D()
        self.loss_D.backward()
        self.optimizer_D.step()


    def compute_loss_G(self):
        super(L2GDLDiscTrainingEnvironment, self).compute_loss_G()

        # Combine batch and time dimensions, but group elements from same time together; then map image intensities
        # to [0, 1]
        _, _, c_dim, H, W = self.gt_middle_frames.shape
        gt_m_frames = inverse_transform(self.gt_middle_frames.permute((1, 0, 2, 3, 4)).contiguous()
                                        .view(-1, c_dim, H, W))

        # Compute reconstruction losses associated with the final prediction
        # Map image intensities to [0, 1]
        outputs = inverse_transform(self.gen_output['pred'].permute(1, 0, 2, 3, 4).contiguous().view(-1, c_dim, H, W))
        self.Lp = self.loss_Lp(outputs, gt_m_frames)
        self.gdl = self.loss_gdl(outputs, gt_m_frames)

        # Compute adversarial loss
        input_fake = torch.cat([self.preceding_frames, self.gen_output['pred'], self.following_frames], dim=1)
        h = self.discriminator(input_fake)
        labels = Variable(torch.ones(h.size())).cuda()
        self.L_GAN = self.loss_d(h, labels)

        self.loss_G += self.alpha * (self.Lp + self.gdl) + self.beta * self.L_GAN


    def get_current_errors(self):
        error_dict = super(L2GDLDiscTrainingEnvironment, self).get_current_errors()

        error_dict['G_Lp'] = self.Lp.data[0]
        error_dict['G_gdl'] = self.gdl.data[0]
        error_dict['D_real'] = self.loss_d_real.data[0]
        error_dict['D_fake'] = self.loss_d_fake.data[0]
        error_dict['G_GAN'] = self.L_GAN.data[0]

        return error_dict


    def train(self):
        """Sets the generator and discriminator models to training mode (e.g. affects dropout and batch-norm layers)."""
        super(L2GDLDiscTrainingEnvironment, self).train()
        self.discriminator.train()


class MCNetTrainingEnvironment(L2GDLDiscTrainingEnvironment):

    def sample_KTF(self, allow_random_sampling):
        if allow_random_sampling:
            K = np.random.randint(2, self.max_K+1)
            T = np.random.randint(1, self.max_T+1)
            F = np.random.randint(1, self.max_F+1)
        else:
            K = self.max_K
            T = self.max_T
            F = self.max_F

        return K, T, F


class TAITrainingEnvironment(L2GDLDiscTrainingEnvironment):

    def sample_KTF(self, allow_random_sampling):
        if allow_random_sampling:
            K = np.random.randint(2, self.max_K+1)
            T = np.random.randint(1, self.max_T+1)
            F = np.random.randint(2, self.max_F+1)
        else:
            K = self.max_K
            T = self.max_T
            F = self.max_F

        return K, T, F

    def compute_loss_G(self):
        """Compute the generator's loss."""

        # Compute the sum of losses on the final prediction
        super(TAITrainingEnvironment, self).compute_loss_G()

        # Combine batch and time dimensions, but group elements from same time together; then map image intensities
        # to [0, 1]
        _, _, c_dim, H, W = self.gt_middle_frames.shape
        gt_m_frames = inverse_transform(self.gt_middle_frames.permute((1, 0, 2, 3, 4)).contiguous()
                                        .view(-1, c_dim, H, W))

        # Map image intensities to [0, 1]
        outputs_forward = inverse_transform(self.gen_output['pred_forward'].permute(1, 0, 2, 3, 4).contiguous()
                                            .view(-1, c_dim, H, W))
        outputs_backward = inverse_transform(self.gen_output['pred_backward'].permute(1, 0, 2, 3, 4).contiguous()
                                             .view(-1, c_dim, H, W))

        # Compute reconstruction losses associated with the intermediate predictions
        self.Lp_forward = self.loss_Lp(outputs_forward, gt_m_frames)
        self.Lp_backward = self.loss_Lp(outputs_backward, gt_m_frames)
        self.gdl_forward = self.loss_gdl(outputs_forward, gt_m_frames)
        self.gdl_backward = self.loss_gdl(outputs_backward, gt_m_frames)

        self.loss_G += self.alpha * (self.Lp_forward + self.Lp_backward + self.gdl_forward + self.gdl_backward)

    def get_current_errors(self):
        """Obtain a dict that specifies the losses associated with training."""

        tai_error_dict = dict([
            ('G_Lp_forward', self.Lp_forward.data[0]),
            ('G_gdl_forward', self.gdl_forward.data[0]),
            ('G_Lp_backward', self.Lp_backward.data[0]),
            ('G_gdl_backward', self.gdl_backward.data[0]),
        ])

        ret = merge_dicts(super(TAITrainingEnvironment, self).get_current_errors(), tai_error_dict)

        return ret

    def get_current_visuals(self):
        """Obtain a dict of video tensors to visualize in TensorBoard."""

        pred_forward_t = self.gen_output['pred_forward']
        pred_backward_t = self.gen_output['pred_backward']

        pred_forward_vis_seq = torch.cat([self.preceding_frames, pred_forward_t, self.following_frames], dim=1)
        pred_backward_vis_seq = torch.cat([self.preceding_frames, pred_backward_t, self.following_frames], dim=1)

        tai_vis_dict = OrderedDict([
            ('pred_forward_vis_seq', pred_forward_vis_seq),
            ('pred_backward_vis_seq', pred_backward_vis_seq),
        ])

        ret = merge_dicts(tai_vis_dict, super(TAITrainingEnvironment, self).get_current_visuals())

        return ret


class SequentialConvTransformerTrainingEnvironment(L2GDLDiscTrainingEnvironment):

    def __init__(self, fill_in_model, checkpoints_dir, name, image_size, c_dim, alpha, beta, lr, beta1, Ip,
                 disc_window_size, num_blocks, num_heads, d_v, d_ff, max_K, max_T, max_F, tf_p_min, tf_p_max,
                 tf_offset, tf_decay, padding_size):
        """Constructor

        :param fill_in_model: A tuple of arguments used to initialize the generator associated with this environment
        :param checkpoints_dir: The root folder where checkpoints are stored
        :param name: The name of the experiment
        :param image_size: The spatial resolution of the video
        :param c_dim: The number of color channels (e.g. 3 for RGB)
        :param alpha: The weight of the image reconstruction-based loss
        :param beta: The weight of the adversarial/discriminator-based loss
        :param lr: The learning rate of the optimizers
        :param beta1: The first beta term used by the Adam optimizer
        :param df_dim: Controls the number of features in each layer of the discriminator
        :param Ip: The number of power iterations to use when computing max singular value (used if sn is True)
        :param disc_t: The total number of frames per video that the discriminator will take in
        :param tf_rate: The percentage of forward passes that should be computed with teacher-forcing
        """
        super(SequentialConvTransformerTrainingEnvironment, self).__init__(fill_in_model, checkpoints_dir, name,
                                                                           image_size, c_dim, alpha, beta, lr, beta1,
                                                                           256, Ip, disc_window_size, max_K, max_T,
                                                                           max_F, padding_size)

        # Check teacher-forcing rate function parameters
        assert(0 <= tf_p_min <= 1 and 0 <= tf_p_max <= 1)
        assert(tf_p_max >= tf_p_min)
        # Construct teacher-forcing rate function
        sigmoid = lambda x: 1. / (1 + np.exp(-x))  # Numerically stable if x > 0
        self.tf_rate_fn = lambda x: (tf_p_min - tf_p_max) * sigmoid((x - tf_offset) / tf_decay) + tf_p_max


    def forward_train(self):
        use_tf_flag = np.random.uniform() < self.tf_rate_fn(self.total_updates)
        if use_tf_flag:
            self.gen_output = self.generator.forward_train(self.preceding_frames, self.gt_middle_frames,
                                                           self.following_frames)
        else:
            self.gen_output = self.generator.forward(self.T, self.preceding_frames, self.following_frames)


class SloMoTrainingEnvironment(BaseTrainingEnvironment):

    def __init__(self, fill_in_model, checkpoints_dir, name, lr, beta1, max_K, max_T, max_F, padding_size, lambda_r, lambda_p, lambda_w, lambda_s,
                 lr_decay_count, lr_decay_rate):
           
        super(SloMoTrainingEnvironment, self).__init__(fill_in_model, checkpoints_dir, name, lr, beta1, max_K, max_T,
                                                       max_F, padding_size)
        self.l1_loss = torch.nn.L1Loss(size_average=True, reduce=True)
        self.MSE_loss = torch.nn.MSELoss(size_average=True, reduce=True)
        self.gdl = GDL()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv = torch.nn.Sequential(*list(torch.nn.Sequential(*list(vgg16.children())[0]))[:22]).cuda()
        for param in self.vgg16_conv.parameters():
            param.requires_grad = False
        self.warper = FlowWarper()
        self.lambda_r = lambda_r
        self.lambda_p = lambda_p
        self.lambda_w = lambda_w
        self.lambda_s = lambda_s
        self.lr_decay_count = lr_decay_count
        self.lr_decay_rate = lr_decay_rate
        self.lr = lr

    def compute_loss_G(self):
        """Compute the generator's loss."""
        super(SloMoTrainingEnvironment, self).compute_loss_G()

        # Combine batch and time dimensions, but group elements from same time together; then map image intensities
        # to [0, 1]
        B, T, c_dim, H, W = self.gt_middle_frames.shape
        I0 = self.preceding_frames[:, -1, :, :, :]
        I1 = self.following_frames[:, 0, :, :, :]
        pred = self.gen_output["pred"]
        F_0_1 = self.gen_output["F_0_1"]
        F_1_0 = self.gen_output["F_1_0"]
        F_t_0_collector = self.gen_output["F_t_0_collector"]
        F_t_1_collector = self.gen_output["F_t_1_collector"]

        # reconstruction loss
        self.reconstruction_loss = self.l1_loss(pred, self.gt_middle_frames)

        # perceptual loss
        vgg16_pred_input = pred.expand(B, T, 3, H, W) if c_dim == 1 else pred
        vgg16_preds = [self.vgg16_conv(vgg16_pred_input[:, i, :, :, :]) for i in range(T)]
        vgg16_preds_new = [p.unsqueeze(1) for p in vgg16_preds]
        vgg16_pred = torch.cat(vgg16_preds_new, 1)
        vgg16_truth_input = self.gt_middle_frames.expand(B, T, 3, H, W) if c_dim == 1 else self.gt_middle_frames
        vgg16_truths = [self.vgg16_conv(vgg16_truth_input[:, i, :, :, :]) for i in range(T)]
        vgg16_truths_new = [p.unsqueeze(1) for p in vgg16_truths]
        vgg16_truth = torch.cat(vgg16_truths_new, 1)
        self.perceptual_loss = self.MSE_loss(vgg16_pred, vgg16_truth)

        # warping loss
        warping_loss_vector = [self.l1_loss(self.warper(I0, F_t_0_collector[:, i, :, :, :]), self.gt_middle_frames[:, i, :, :, :]) + 
                               self.l1_loss(self.warper(I1, F_t_1_collector[:, i, :, :, :]), self.gt_middle_frames[:, i, :, :, :]) for i in range(self.gt_middle_frames.shape[1])]
        self.warping_loss = self.l1_loss(self.warper(I0, F_1_0), I1) + self.l1_loss(self.warper(I1, F_0_1), I0) + sum(warping_loss_vector) / len(warping_loss_vector)

        # smoothness loss
        smooth_loss_1_0 = self.gdl(F_1_0.contiguous(), as_variable(torch.zeros(F_1_0.shape)).cuda())
        smooth_loss_0_1 = self.gdl(F_0_1.contiguous(), as_variable(torch.zeros(F_0_1.shape)).cuda())
        self.smooth_loss = smooth_loss_1_0 + smooth_loss_0_1

        # overall loss
        self.loss_G += self.lambda_r * self.reconstruction_loss + self.lambda_p * self.perceptual_loss + self.lambda_w * self.warping_loss + self.lambda_s * self.smooth_loss

    def get_current_errors(self):
        """Obtain a list that specifies the losses associated with training."""

        error_dict = super(SloMoTrainingEnvironment, self).get_current_errors()

        error_dict['reconstruction_loss'] = self.reconstruction_loss.data[0]
        error_dict['perceptual_loss'] = self.perceptual_loss.data[0]
        error_dict['warping_loss'] = self.warping_loss.data[0]
        error_dict['smooth_loss'] = self.smooth_loss.data[0]

        return error_dict

    def optimize_parameters(self):

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = self.lr * (self.lr_decay_rate ** (self.total_updates // self.lr_decay_count))
        self.optimizer_G.zero_grad()
        self.compute_loss_G()
        self.loss_G.backward()
        self.optimizer_G.step()
