# A Temporally-Aware Interpolation Network for Video Frame Inpainting
Ximeng Sun, Ryan Szeto and Jason J. Corso, University of Michigan

**Note:** This is a fork of [this code base](https://github.com/sunxm2357/TAI_video_frame_inpainting). The original code implemented the experiments in the ACCV 2018 paper "A Temporally-Aware Interpolation Network for Video Frame Inpainting".


## Setup the environment
In this section, we provide an instruction to help you set up the experiment environment. This code was tested for Python 2.7 on Ubuntu 16.04, and is based on PyTorch 0.3.1.

First, please make sure you have installed  `cuda(8.0.61)`, `cudnn(8.0-v7.0.5)` `opencv(2.4.13)`, and `ffmpeg(2.2.3)`.  Note that different versions might not work.

Then, to avoid conflicting with your current environment, we suggest to use `virtualenv`. For using `virtualenv`, please refer https://virtualenv.pypa.io/en/stable/. Below is an example of initializing and activating a virtual environment:

```bash
virtualenv .env
source .env/bin/activate
```

After activating a virtual environment, use the following command to install all dependencies you need to run this model.

```bash
pip install -r requirements.txt
```

### Compiling the separable convolution module

The separable convolution module uses CUDA code that must be compiled before it can be used. To compile, start by activating the virtual environment described above. Then, set the `CUDA_HOME` environment variable as the path to your CUDA installation. For example, if your CUDA installation is at `/usr/local/cuda-8.0`, run this command:

```bash
export CUDA_HOME=/usr/local/cuda-8.0
```

Then, identify a virtual architecture that corresponds to your GPU from [this site](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) (e.g. `compute_52` for Maxwell GPUs). Finally, to compile the CUDA code, run the install script with the virtual architecture as an argument, e.g.:

```bash
bash bashes/misc/install.bash compute_52
```


## Download datasets

In our paper, we use the KTH Actions, UCF-101, and HMDB-51 datasets to evaluate our method. Use the commands below to download these datasets. The sample commands below download the datasets into a new `datasets` directory, but you can replace this argument with another location if needed.


```bash
bash bashes/download/download_KTH.bash datasets
bash bashes/download/download_UCF.bash datasets
bash bashes/download/download_HMDB.bash datasets
bash bashes/download/download_Imagenet.bash datasets
```

**Note: If you use a location other than `datasets`, change the files in `videolist` to point to the correct video paths. Alternatively, you can create a symlink from `datasets` to your custom folder.**

**Note: Each script may take about 1.5 hours on a 1 Gbit/s connection, but they can be run in parallel.**

## Train our model and baselines

### Train single model

The script `bashes/experiments/train.sh` runs a training script whose options are determined by a *default arguments* file and an *extra arguments* file, which are located under `exp_args/default_args` and `exp_args/extra_args` respectively. In general, the default arguments define experimental settings that should not change between different models (such as the set of video clips or the number of color channels), and extra arguments specify settings that do change between models (e.g. the experiment name or network architecture), or between training and testing time (e.g. whether to pad the images at inference time).

To run the `train.sh` script, call it from the project root directory, and pass in the paths to the default and extra argument files. For example:

```
./bashes/experiments/train.sh exp_args/default_args/KTH/train.txt exp_args/extra_args/KTH/bi-TAI.txt
```

*Note: For UCF-101 and HMDB-51, Super SloMo uses different extra argument files between training and testing time. In these cases, you should use the `SuperSloMo_train.txt` file for the given dataset.*

Training takes a long time (our bi-TAI models took 70 hours on a Xeon E5-2640 CPU and one Titan Xp GPU).

### train.py

Alternatively, you can run `train.py` directly while specifying all command-line arguments. For instance, the above command expands out to:

```
python train.py \
    --K=5 \
    --F=5 \
    --T=5 \
    --alt_T=10 \
    --alt_K=7 \
    --alt_F=7 \
    --max_iter=200000 \
    --train_video_list_path=videolist/KTH/train_data_list.txt \
    --val_video_list_path=videolist/KTH/val_data_list_T=5.txt \
    --val_video_list_alt_T_path=videolist/KTH/val_data_list_T=10.txt \
    --val_video_list_alt_K_F_path=videolist/KTH/val_data_list_K=F=7.txt \
    --vis_video_list_path=videolist/KTH/vis_data_list_T=5.txt \
    --vis_video_list_alt_T_path=videolist/KTH/vis_data_list_T=10.txt \
    --vis_video_list_alt_K_F_path=videolist/KTH/vis_data_list_K=F=7.txt \
    --c_dim=1 \
    --image_size=128 \
    --sample_KTF \
    --name=kth_bi-TAI \
    --model_key=TAI_gray
```

`train.py` takes arguments that define the name of the experiment, the model to be used, the dataset location, and other information. To read about all of the arguments that are supported during training, run

```bash
python train.py --help
```


### Visualize the training process

Losses and visualizations are saved in a TensorBoard file under `tb` for each experiment. In order to view these intermediate results, you can activate a TensorBoard instance with `tb` as the log directory:

```bash
tensorboard --logdir tb
```


## Evaluate models

### Downloading pre-trained models

We have trained our models on the KTH Actions, UCF-101, and HMDB-51 action datasets and made the snapshots publicly available. To extract them to the `checkpoints` folder, run the following command (WARNING: This will overwrite any models that you have trained using our provided default and extra argument files):

```
bash bashes/download/download_model_checkpoints.bash
```

### Evaluate single model

Before you start, make sure you have trained or downloaded the model that you want to evaluate.

To evaluate each model on its held-out test set, you first need to save the model's predictions to disk. To do this, run `bashes/experiments/predict.sh` from the project root directory, which takes the paths to the default and extra files, as well as the path to the folder where all images should be saved. For consistency, we recommend specifying this folder as `results/<dataset_label>/images/<exp_name>`. Here is an example of running the bi-TAI model on the KTH (m=10) test set:

```
./bashes/experiments/predict.sh exp_args/default_args/KTH/test_10.txt exp_args/extra_args/bi-TAI.txt results/KTH-test_data_list_T=10/images/bi-TAI
```

Next, you need to compute quantitative metrics on the predictions. To do this, run `./bashes/experiments/compute_summarize_quant_results.sh`, which takes four arguments:

1. The root path of the images
2. The path in which to store the quantitative results (in a file called `results.npz`)
3. The number of preceding/following frames
4. The number of middle frames

For example, to compute results on the KTH (m=10) test set (with five preceding and five following frames), run:

```
./bashes/experiments/compute_summarize_quant_results.sh results/KTH-test_data_list_T=10/images/bi-TAI results/KTH-test_data_list_T=10/quantitative/bi-TAI 5 10
```

This script additionally generates plots and other files to summarize the quantitative results in a more human-friendly way. It takes about 1/2-2 hours to finish.

*Note: For UCF-101 and HMDB-51, Super SloMo uses different extra argument files between training and testing time. In these cases, you should use the `SuperSloMo_val_test.txt` file for the given dataset.*

#### Newson et al.

To generate predictions from Newson et al., we used their publicly-available code ([link](https://perso.telecom-paristech.fr/gousseau/video_inpainting/)). We have provided the outputs of their method in a ~20GB split `tar.gz` file. To download and extract them to the `results` folder, run the following command:

```
bash bashes/download/download_newson_results.bash
```

Afterwards, run `./bashes/experiments/compute_summarize_quant_results.sh` as described above. For example, on the KTH (m=10) test set:

```
./bashes/experiments/compute_summarize_quant_results.sh results/KTH-test_data_list_T=10/images/Newson results/KTH-test_data_list_T=10/quantitative/Newson 5 10
```

#### More details

In the project root, there are several scripts to help process, evaluate, and visualize predictions (some of them are called by `compute_summarize_quant_results.sh`). Below is a summary of some of them:

1. Prediction/inference (`predict.py`): Use the trained model to generate middle frames
2. Computing quantitative results (`compute_quant_results.py`): Compute difference metrics between the generated and the ground-truth middle frames
3. Summarizing quantitative results (`summarize_quant_results.py`): Produce plots or text files that describe the quantitative results in different ways, such as plotting performance per time step, reporting per-video performance, visualizing the distribution of per-video performance with boxplots, or summarizing the model's performance with one number
4. Generating GIFs (`animate_qual_results.py`): Create animated GIFs of the ground-truth and predicted segments of the video
5. Comparing results between models (`compare_visual_results.py`): Generate images or videos that show the predictions of different models side-by-side.

For an explanation of the arguments supported by the Python scripts listed above, call them with the `--help` flag, e.g.

```
python predict.py --help
```

## Generate figures and tables from the paper

We have provided scripts to generate the figures in our paper. This is a convenient way of checking that your results match ours. To do this, perform the following steps:

1. [Set up the environment](#setup-the-environment)
2. [Download the datasets](#download-datasets)
3. [Download the pre-trained models](#downloading-pre-trained-models)
4. [Download the predictions from Newson et al.](#newson-et-al)
5. [Evaluate the PyTorch models](#evaluate-single-model)
6. [Evaluate the Newson et al. method](#newson-et-al)
7. Run the following lines:

    ```
    ./bashes/evaluation/paper/quantitative.sh
    ./bashes/evaluation/paper/qualitative.sh
    ./bashes/evaluation/supplementary/qualitative.sh
    ```

8. Make sure that the contents of `quant_tables` and `quant_tables_orig` are identical:

    ```
    diff quant_tables_orig quant_tables
    ```

9. Make sure that the contents of `paper_figs` and `paper_figs_orig` are visually identical. Unfortunately, this is the easiest way to compare the PDFs since they're binary files.
10. Make sure that the contents of `supplementary` and `supplementary_orig` are visually identical.

## Predict middle frames for a new video

The `predict.py` script allows you to generate predictions on new videos. Assuming that your video collection is on disk, generating the predictions involves two steps:

1. Construct a video list file
2. Run `predict.py` with a specific set of arguments

This documentation will cover the case in which you interpolate between two contiguous sequences of frames in each video without evaluation.

### Constructing the video list file

The video list file is a `.txt` file where each line defines two contiguous sequences of frames from one video. For example, one line might look like this:

```
/home/user/Videos/test.mp4 1-5 16-20
```

Note that there are three space-separated items in each line. The first item is the path to the video file (the path should either be absolute or relative to the running path). The second and third items specify the index range of the preceding and following frames, respectively. The ranges are 1-indexed and inclusive (so in the example above, the preceding frames are the first through fifth frames from `test.mp4`).

Note: Every line must specify the same number of preceding frames and the same number of following frames (but these two numbers can be different, e.g. five preceding and three following frames).

### Running `predict.py` for inference

After you create the video list file, run `predict.py`. Below is an example:

```bash
python predict.py \
    --name=ucf_TAI_exp1 \
    --K=5 \
    --T=5 \
    --F=5 \
    --c_dim=3 \
    --image_size 240 320 \
    --batch_size=2 \
    --test_video_list_path=/path/to/video_list.txt \
    --model_key=TAI_color \
    --qual_result_root=/path/to/results/root \
    --disjoint_clips
```

This is what each argument specifies:

* name: The name of the trained model. This corresponds to the folder names under `checkpoints`.
* K, F: Specifies the number of preceding and following frames. Make sure these match the number of frames you specified in the video text file.
* T: Specifies the number of middle frames to generate.
* c_dim: The number of color channels in the resulting video (1 for grayscale, 3 for RGB). If you're using a trained model like TAI, this must match the number of channels that was used to train that model.
* image_size: The height and width of the frames to produce.
* batch_size: The number of videos to process in one forward pass. Generally, this should be as high as possible without making your computer run out of GPU or CPU memory.
* test_video_list_path: The path to your video list file.
* model_key: The name of the architecture or method to generate video frames with. Must match the value used to train the model.
* qual_result_root: The path under which to store all generated frames.
* disjoint_clips: This flag indicates that you want to do inference on two contiguous sequences of frames. Note that when this flag is on, ground-truth middle frames will not be produced (because in general, there is no ground-truth middle sequence that corresponds to the given disjoint clips).

Note: For a trained model (e.g. TAI), the script will use the best checkpoint by default (i.e. the one under `checkpoints/<name>/model_best.ckpt`). You can specify another checkpoint by using the `snapshot_file_name` argument.

After running `predict.py`, each video clip will have its own folder under `qual_result_root`. These folders will contain raw images for the input and and predicted frames, from which you can use other tools for visualization (e.g. `animate_qual_results.py` or an external program).
