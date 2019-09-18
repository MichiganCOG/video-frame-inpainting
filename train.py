import os
import time

import numpy as np
import skimage.measure as measure
import torch
from skimage.measure import compare_ssim as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from src.data.base_dataset import ContiguousVideoClipDataset
from src.environments.environments import create_training_environment
from src.models.create_model import create_model
from src.options.options import TrainOptions
from src.util.util import makedir, listopt, visual_grid, to_numpy, inverse_transform, draw_err_plot, inf_data_loader,\
    dict_to_markdown_table_str


def main():
    opt = TrainOptions().parse()

    # Make folders for this experiment
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    makedir(expr_dir)
    tb_dir = os.path.join(opt.tensorboard_dir, opt.name)
    makedir(tb_dir)

    # Print options to console and to a file
    listopt(opt)
    with open(os.path.join(expr_dir, 'train_opt.txt'), 'wt') as opt_file:
        listopt(opt, opt_file)
    # Print options to TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    writer.add_text('args', dict_to_markdown_table_str(vars(opt)))

    # Create training datasets
    train_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.train_video_list_path, opt.K + opt.T + opt.F,
                                               not opt.no_backwards, not opt.no_flip, opt.image_size, True,
                                               opt.padding_size)
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, drop_last=True)
    inf_train_data_loader = inf_data_loader(train_data_loader)
    print('# training videos = %d' % len(train_dataset))

    # Create validation dataset (with train number of middle frames)
    val_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.val_video_list_path, opt.K + opt.T + opt.F, False, False,
                                             opt.image_size, False, opt.padding_size)
    val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads,
                                 drop_last=False)
    print('# val videos = %d' % len(val_dataset))

    # Create validation dataset (with alt number of middle frames)
    val_alt_T_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.val_video_list_alt_T_path, opt.K + opt.alt_T + opt.F,
                                                   False, False, opt.image_size, False, opt.padding_size)
    val_alt_T_data_loader = DataLoader(val_alt_T_dataset, batch_size=opt.batch_size, shuffle=False,
                                       num_workers=opt.num_threads, drop_last=False)
    print('# val (alt T) videos = %d' % len(val_alt_T_dataset))

    # Create validation dataset (with alt number of preceding and following frames)
    val_alt_K_F_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.val_video_list_alt_K_F_path,
                                                     opt.alt_K + opt.T + opt.alt_F, False, False, opt.image_size, False,
                                                     opt.padding_size)
    val_alt_K_F_data_loader = DataLoader(val_alt_K_F_dataset, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=opt.num_threads, drop_last=False)
    print('# val (alt K/F) videos = %d' % len(val_alt_K_F_dataset))

    # Create visualization dataset (with train number of middle frames)
    vis_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.vis_video_list_path, opt.K + opt.T + opt.F, False, False,
                                             opt.image_size, False, opt.padding_size)
    vis_data_loader = DataLoader(vis_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads,
                                 drop_last=False)
    print('# visualization videos = %d' % len(vis_dataset))

    # Create visualization dataset (with alt number of middle frames)
    vis_alt_T_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.vis_video_list_alt_T_path, opt.K + opt.alt_T + opt.F,
                                                   False, False, opt.image_size, False, opt.padding_size)
    vis_alt_T_data_loader = DataLoader(vis_alt_T_dataset, batch_size=opt.batch_size, shuffle=False,
                                       num_workers=opt.num_threads, drop_last=False)
    print('# visualization (alt T) videos = %d' % len(vis_alt_T_dataset))

    # Create visualization dataset (with alt number of preceding and following frames)
    vis_alt_K_F_dataset = ContiguousVideoClipDataset(opt.c_dim, opt.vis_video_list_alt_K_F_path,
                                                     opt.alt_K + opt.T + opt.alt_F, False, False, opt.image_size, False,
                                                     opt.padding_size)
    vis_alt_K_F_data_loader = DataLoader(vis_alt_K_F_dataset, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=opt.num_threads, drop_last=False)
    print('# visualization (alt K/F) videos = %d' % len(vis_alt_K_F_dataset))

    # Create model and training environment
    fill_in_model = create_model(opt.model_key)
    env = create_training_environment(fill_in_model, opt.c_dim, opt.checkpoints_dir, opt.name, opt.K, opt.T, opt.F,
                                      opt.image_size, opt.alpha, opt.beta, opt.lr, opt.beta1, opt.df_dim, opt.Ip,
                                      opt.disc_window_size, opt.tf_p_min, opt.tf_p_max, opt.tf_offset, opt.tf_decay,
                                      opt.padding_size, opt.lambda_r, opt.lambda_p, opt.lambda_w, opt.lambda_s,
                                      opt.lr_decay_count, opt.lr_decay_rate)

    total_updates = env.start_update
    env.total_updates = total_updates
    best_sum_avg_psnr_err = env.start_sum_avg_psnr_err
    best_sum_avg_ssim_err = env.start_sum_avg_ssim_err

    for data in inf_train_data_loader:
        iter_start_time = time.time()
        total_updates += 1
        env.total_updates = total_updates

        # Sample K, T, F if random sampling is allowed (otherwise, always sample K, T, F specified by cmd line args)
        K, T, F = env.sample_KTF(opt.sample_KTF)

        # Update model
        all_frames = data['targets']
        preceding_frames = all_frames[:, :K, :, :, :]
        gt_middle_frames = all_frames[:, K:K+T, :, :, :]
        following_frames = all_frames[:, K+T:K+T+F, :, :, :]
        env.set_train_inputs(preceding_frames, following_frames, gt_middle_frames)
        env.K, env.T, env.F = K, T, F
        env.train()
        env.forward_train()
        env.optimize_parameters()

        if total_updates % opt.print_freq == 0:
            error_dict = env.get_current_errors()
            time_per_item = (time.time() - iter_start_time) / opt.batch_size
            for key, value in error_dict.iteritems():
                writer.add_scalar('loss/%s' % key, value, total_updates)
            writer.add_scalar('iter_time', time_per_item, total_updates)
            # Write to console
            message = 'total_updates: %d, time_per_item: %.3f ' % (total_updates, time_per_item)
            for k, v in error_dict.iteritems():
                message += '%s: %.3f ' % (k, v)
            print(message)
            # Draw visual results in TensorBoard
            visuals = env.get_current_visuals()
            vis_grid = visual_grid(visuals, K, T)
            writer.add_image('current_batch', vis_grid, total_updates)

        if total_updates % opt.save_latest_freq == 0:
            print('saving the latest model (update %d)' % total_updates)
            env.save('model_latest.ckpt', total_updates, best_sum_avg_psnr_err, best_sum_avg_ssim_err)
            env.save('model_%08d.ckpt' % total_updates, total_updates, best_sum_avg_psnr_err, best_sum_avg_ssim_err)

        if total_updates % opt.validate_freq == 0:

            ### VALIDATE WITH TRAINING NUMBER OF FRAMES ###
            start_time = time.time()
            psnr_err, ssim_err, l2_err = compute_errors(env, val_data_loader, opt.c_dim != 1, opt.K, opt.T, opt.F)
            elapsed_seconds = time.time() - start_time
            print('Validation (T=%d) done. Took %.03f minutes' % (opt.T, elapsed_seconds / 60))

            # Write image quality scores to TensorBoard
            writer.add_scalar('loss_val/psnr/T=%d' % opt.T, psnr_err.mean(), total_updates)
            writer.add_scalar('loss_val/ssim/T=%d' % opt.T, ssim_err.mean(), total_updates)
            writer.add_scalar('loss_val/G_Lp/T=%d' % opt.T, l2_err.mean(), total_updates)

            psnr_plot = draw_err_plot(psnr_err, 'Peak Signal to Noise Ratio', [1, opt.T, 0, 35])
            ssim_plot = draw_err_plot(ssim_err, 'Structural Similarity', [1, opt.T, 0, 1])
            writer.add_image('psnr/T=%d' % opt.T, psnr_plot, total_updates)
            writer.add_image('ssim/T=%d' % opt.T, ssim_plot, total_updates)
            vis_grid = visualize_predictions(env, vis_data_loader, opt.K, opt.T, opt.F)
            writer.add_image('samples/T=%d' % opt.T, vis_grid, total_updates)

            # If current snapshot has best performance, save it and update the best PSNR/SSIM value
            sum_avg_psnr_err = np.sum(np.mean(psnr_err, axis=0))
            sum_avg_ssim_err = np.sum(np.mean(ssim_err, axis=0))
            if sum_avg_ssim_err > best_sum_avg_ssim_err:
                print('Current model has best SSIM, saving...')
                env.save('model_best.ckpt', total_updates, sum_avg_psnr_err, sum_avg_ssim_err)
                best_sum_avg_psnr_err = sum_avg_psnr_err
                best_sum_avg_ssim_err = sum_avg_ssim_err

            ### VALIDATE WITH ALTERNATIVE NUMBER OF MIDDLE FRAMES ###
            start_time = time.time()
            psnr_err_alt_T, ssim_err_alt_T, l2_err_alt_T = compute_errors(env, val_alt_T_data_loader, opt.c_dim != 1,
                                                                          opt.K, opt.alt_T, opt.F)
            elapsed_seconds = time.time() - start_time
            print('Validation (T=%d) done. Took %.03f minutes' % (opt.alt_T, elapsed_seconds / 60))

            # Write image quality scores to TensorBoard
            writer.add_scalar('loss_val/psnr/T=%d' % opt.alt_T, psnr_err_alt_T.mean(), total_updates)
            writer.add_scalar('loss_val/ssim/T=%d' % opt.alt_T, ssim_err_alt_T.mean(), total_updates)
            writer.add_scalar('loss_val/G_Lp/T=%d' % opt.alt_T, l2_err_alt_T.mean(), total_updates)

            psnr_plot_alt_T = draw_err_plot(psnr_err_alt_T, 'Peak Signal to Noise Ratio', [1, opt.alt_T, 0, 35])
            ssim_plot_alt_T = draw_err_plot(ssim_err_alt_T, 'Structural Similarity', [1, opt.alt_T, 0, 1])
            writer.add_image('psnr/T=%d' % opt.alt_T, psnr_plot_alt_T, total_updates)
            writer.add_image('ssim/T=%d' % opt.alt_T, ssim_plot_alt_T, total_updates)
            vis_grid_alt_T = visualize_predictions(env, vis_alt_T_data_loader, opt.K, opt.alt_T, opt.F)
            writer.add_image('samples/T=%d' % opt.alt_T, vis_grid_alt_T, total_updates)

            ### VALIDATE WITH ALTERNATIVE NUMBER OF PRECEDING/FOLLOWING FRAMES ###
            start_time = time.time()
            psnr_err_alt_K_F, ssim_err_alt_K_F, l2_err_alt_K_F = compute_errors(env, val_alt_K_F_data_loader,
                                                                                opt.c_dim != 1, opt.alt_K, opt.T,
                                                                                opt.alt_F)
            elapsed_seconds = time.time() - start_time
            print('Validation (K=%d, F=%d) done. Took %.03f minutes' % (opt.alt_K, opt.alt_F, elapsed_seconds / 60))

            # Write image quality scores to TensorBoard
            writer.add_scalar('loss_val/psnr/K=%d_F=%d' % (opt.alt_K, opt.alt_F), psnr_err_alt_K_F.mean(), total_updates)
            writer.add_scalar('loss_val/ssim/K=%d_F=%d' % (opt.alt_K, opt.alt_F), ssim_err_alt_K_F.mean(), total_updates)
            writer.add_scalar('loss_val/G_Lp/K=%d_F=%d' % (opt.alt_K, opt.alt_F), l2_err_alt_K_F.mean(), total_updates)

            psnr_plot_alt_K_F = draw_err_plot(psnr_err_alt_K_F, 'Peak Signal to Noise Ratio', [1, opt.T, 0, 35])
            ssim_plot_alt_K_F = draw_err_plot(ssim_err_alt_K_F, 'Structural Similarity', [1, opt.T, 0, 1])
            writer.add_image('psnr/K=%d_F=%d' % (opt.alt_K, opt.alt_F), psnr_plot_alt_K_F, total_updates)
            writer.add_image('ssim/K=%d_F=%d' % (opt.alt_K, opt.alt_F), ssim_plot_alt_K_F, total_updates)
            vis_grid_alt_K_F = visualize_predictions(env, vis_alt_K_F_data_loader, opt.alt_K, opt.T, opt.alt_F)
            writer.add_image('samples/K=%d_F=%d' % (opt.alt_K, opt.alt_F), vis_grid_alt_K_F, total_updates)

        if total_updates >= opt.max_iter:
            env.save('model_latest.ckpt', total_updates, best_sum_avg_psnr_err, best_sum_avg_ssim_err)
            break


def visualize_predictions(env, data_loader, K, T, F):
    # Generate the fill-in predictions and store the visual results
    vis = []
    for d in data_loader:
        all_frames = d['targets']
        preceding_frames = all_frames[:, :K, :, :, :]
        gt_middle_frames = all_frames[:, K:-F, :, :, :]
        following_frames = all_frames[:, -F:, :, :, :]
        env.set_test_inputs(preceding_frames, following_frames)
        env.K, env.T, env.F = K, T, F
        env.eval()
        env.forward_test()

        # Set environment's GT middle frames, which are used for visualization
        env.set_gt_middle_frames_test(gt_middle_frames)
        val_visuals = env.get_current_visuals()
        batch_vis_grid = visual_grid(val_visuals, K, T)
        vis.append(batch_vis_grid)
    val_vis_grid = torch.cat(vis, dim=1)
    return val_vis_grid


def compute_errors(env, data_loader, multichannel, K, T, F):
    # Initialize PSNR and SSIM tables
    psnr_err = np.zeros((0, T))
    ssim_err = np.zeros((0, T))
    l2_err = np.zeros((0, T))

    # Pass through given data
    for d in data_loader:
        # prepare the ground truth in the form of [batch, f, h, w, c]
        # compute the inpainting results
        all_frames = d['targets']
        preceding_frames = all_frames[:, :K, :, :, :]
        gt_middle_frames = all_frames[:, K:-F, :, :, :]
        following_frames = all_frames[:, -F:, :, :, :]
        env.set_test_inputs(preceding_frames, following_frames)
        env.K, env.T, env.F = K, T, F
        env.eval()
        env.forward_test()

        # Concatenate and clip color intensities in predicted and GT frames
        pred_data = to_numpy(env.gen_output['pred'].data, (0, 1, 3, 4, 2))
        true_data = to_numpy(gt_middle_frames, (0, 1, 3, 4, 2)).copy()

        pred_data = pred_data.clip(-1, 1)
        true_data = true_data.clip(-1, 1)

        if not multichannel:
            pred_data = np.squeeze(pred_data, axis=-1)
            true_data = np.squeeze(true_data, axis=-1)

        # Compute SSIM and PSNR curves for each video in current batch
        for b in range(pred_data.shape[0]):
            cpsnr = np.zeros((T,))
            cssim = np.zeros((T,))
            cl2 = np.zeros((T,))
            # Compute SSIM and PSNR for each frame in current video
            for time_per_item in range(T):
                pred_float = inverse_transform(pred_data[b, time_per_item])
                target_float = inverse_transform(true_data[b, time_per_item])
                # Compute L2 error based on float versions of images (range 0-1)
                cl2[time_per_item] = ((pred_float - target_float) ** 2).mean()
                # Compute PSNR and SSIM based on uint8 versions of images
                pred_uint8 = (pred_float * 255).astype('uint8')
                target_uint8 = (target_float * 255).astype('uint8')
                cpsnr[time_per_item] = measure.compare_psnr(pred_uint8, target_uint8)
                cssim[time_per_item] = ssim(target_uint8, pred_uint8, multichannel=multichannel)
            psnr_err = np.concatenate((psnr_err, cpsnr[None, :]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None, :]), axis=0)
            l2_err = np.concatenate((l2_err, cl2[None, :]), axis=0)

    return psnr_err, ssim_err, l2_err


if __name__ == '__main__':
    main()
