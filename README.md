# Semantic-Segmentation-CRF

## Dependencies
---
- Python 3.x
- TensorFlow 1.10.0
- CUDA 9.0

# Get deeplabv3 and deeplabv3+ results
---
I've used three models that are deeplabv3 trained on PASCALVOC2012 train+aug dataset and its backbone is resnet_v2_101、deeplabv3+ trained on PASCAL VOC2012 train+aug and its backbone is xception_65、deeplabv3+ trained on PASCAL VOC2012 train+val and its backbone is Xception_65.To be honest, deeplabv3 is my own work，and I refer to this repo about deeplabv3+:[deeplabv3+](https://github.com/xiaochunWu/models/blob/master/research/deeplab/g3doc/pascal.md)

## PASCAL dataset
---
- [PASCAL VOC training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), specifying the location with '--data_dir'.
- [augmented segmentation label](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) 
(Thanks to DrSleep), specifying the location with `--label_data_dir`.
- [model](https://www.dropbox.com/s/gzwb0d6ydpfoxoa/deeplabv3_ver1.tar.gz?dl=0), specifying the location with
`--model_dir`.
- For training, you need to download and extract 
[pre-trained Resnet v2 101 model](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)
from [slim](https://github.com/tensorflow/models/tree/master/research/slim)
specifying the location with `--pre_trained_model`.
## Deeplabv3
---
### Training
First, we convert original data to the Tensorflow TFRecord format to accelerate training seep.
```bash
python create_tf_record.py --data_dir DATA_DIR \ 
                           --image_data_dir IMAGE_DATA_DIR \
                           --label_data_dir LABEL_DATA_DIR
```
then start training model as follow:
```bash
python train.py --model_dir MODEL_DIR \
                --pre_trained_model PRE_TRAINED_MODEL \ 
                --batch_size 16 \
                --train_epochs 46 \
                --data_dir DATA_DIR
```
MODEL_DIR is the directory contains checkpoints.--batch_size 16 because I use TITAN V(16GB)
### evaluate
To evaluate how model perform, you can do this with saved checkpoints:
```bash
python evaluate.py --image_data_dir IMAGE_DATA_DIR \
                   --label_data_dir IMAGE_DATA_DIR \
                   --evaluation_data_list EVALUATION_DATA_LIST \
                   --model_dir MODEL_DIR
```
The current best model build by this implementation achieves 75.72% mIOU on the PASCAL VOC 2012 test dataset.
I also try to train this model on MS_COCO dataset, respectively used all images and only 21-class images.

|   |Method|Dataset|OS|mIOU|
|:--:|:--:|:--:|:--:|:--:|
|paper|deeplabv3|PASCAL VOC 2012 train    |16|77.21%|
|repo |deeplabv3|PASCAL VOC 2012 train+aug|16|75.72%|
|repo |deeplabv3|MS-COCO 21-class         |16|69.11%|
|repo |deeplabv3|MS-COCO 91-class         |16|56.34%|

### inference
To apply semantic segmentation to your image, you can do as follow:
```bash
python inference.py --data_dir DATA_DIR \
                    --infer_data_list INFER_DATA_LIST \
                    --model_dir MODEL_DIR \
                    --output_dir OUTPUT_DIR
```
## Deeplabv3+
---
### Running the train/eval/vis jobs

A local training job using `xception_65` can be run with the following command:

```bash
# From tensorflow/models/research/
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=1 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

where ${PATH_TO_INITIAL_CHECKPOINT} is the path to the initial checkpoint
(usually an ImageNet pretrained checkpoint), ${PATH_TO_TRAIN_DIR} is the
directory in which training checkpoints and events will be written to, and
${PATH_TO_DATASET} is the directory in which the PASCAL VOC 2012 dataset
resides.

**Note that for {train,eval,vis}.py:**

1.  In order to reproduce our results, one needs to use large batch size (> 12),
    and set fine_tune_batch_norm = True. Here, we simply use small batch size
    during training for the purpose of demonstration. If the users have limited
    GPU memory at hand, please fine-tune from our provided checkpoints whose
    batch norm parameters have been trained, and use smaller learning rate with
    fine_tune_batch_norm = False.

2.  The users should change atrous_rates from [6, 12, 18] to [12, 24, 36] if
    setting output_stride=8.

3.  The users could skip the flag, `decoder_output_stride`, if you do not want
    to use the decoder structure.

A local evaluation job using `xception_65` can be run with the following
command:

```bash
# From tensorflow/models/research/
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --eval_logdir=${PATH_TO_EVAL_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

where ${PATH_TO_CHECKPOINT} is the path to the trained checkpoint (i.e., the
path to train_logdir), ${PATH_TO_EVAL_DIR} is the directory in which evaluation
events will be written to, and ${PATH_TO_DATASET} is the directory in which the
PASCAL VOC 2012 dataset resides.

A local visualization job using `xception_65` can be run with the following
command:

```bash
# From tensorflow/models/research/
python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=513 \
    --vis_crop_size=513 \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --vis_logdir=${PATH_TO_VIS_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

where ${PATH_TO_CHECKPOINT} is the path to the trained checkpoint (i.e., the
path to train_logdir), ${PATH_TO_VIS_DIR} is the directory in which evaluation
events will be written to, and ${PATH_TO_DATASET} is the directory in which the
PASCAL VOC 2012 dataset resides. Note that if the users would like to save the
segmentation results for evaluation server, set also_save_raw_predictions =
True.

**To do stacking, you should use different checkpoints and dataset to get 2 results.**

# Stacking
---
now you get three results, and then do stacking.
The stacking rule is vote, choose the most possible label about every pixel through above three results.
Do as follow:
```bash
python pixel_stacking.py --path_val PATH_VAL \
                         --path_aug PATH_AUG \
                         --path_ori PATH_ORI \
                         --path_stacking PATH_STACKING
```
PATH_VAL、PATH_AUG、PATH_ORI are the directories contains three results, PATH_STACKING is the path to output.

# denseCRF
---
Do as follow, then you can get the result after denseCRF.
```bash
python densecrf_inference.py --image_data_dir IMAGE_DATA_DIR \
                             --label_data_dir LABEL_DATA_DIR \
                             --output_dir OUTPUT_DIR \
                             --test_data_list TEST_DATA_LIST
```
IMAGE_DATA_DIR is the path of original images, LABEL_DATA_DIR is the path of segmentation results.
you can view all results through the following table.

|   |Method|Dataset|OS|mIOU|
|:--:|:--:|:--:|:--:|:--:|
|paper|deeplabv3|PASCAL VOC 2012 train|16|77.21%|
|repo|deeplabv3|PASCAL VOC 2012 train+aug|16|75.72%|
|repo|deeplabv3|MS-COCO 21-class|16|69.11%|
|repo|deeplabv3|MS-COCO 31-class|16|56.34%|
|paper|deeplabv3+|PASCAL VOC 2012 train+aug|16|83.68|
|paper|deeplabv3+|PASCAL VOC 2012 train+val|16|87.8|
|repo|deeplabv3+deeplabv3++deeplabv3+|-|-|88.1|
|repo|deeplabv3+deeplabv3++deeplabv3++denseCRF|-|-|84.12|

You can get more details in my [blog](http://wuxiaochun.cn/2018/11/23/semantic-segmentation-Implementation/#more)

# Acknowledgements
---
- [DrSleep's DeepLab-ResNet (DeepLabv2)](https://github.com/DrSleep/tensorflow-deeplab-resnet)
- [tensorflow](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim) 
