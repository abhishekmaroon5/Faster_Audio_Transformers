# Faster_Audio_Transformers
Merging Tokens to make Audio Transformers faster.


This codebase is inspired from below works:

## https://github.com/YuanGongND/ast
## https://github.com/facebookresearch/ToMe

# Model Details:
**model_size:** 'base384'

## Training for Speech Command:
hi
1. Navigate to the project directory:
    ```bash
    cd /data/swarup_behera/Research/TOME/ToMe/egs/speechcommands
    ```

2. Activate the Conda environment:
    ```bash
    conda activate ast-tome

    or 

    conda env create -f environment.yml
    conda activate ast-tome
    ```

3. Run the training command:
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -W ignore ../../src/run.py --model ast --dataset speechcommands --data-train ./data/datafiles/speechcommand_train_data.json --data-val ./data/datafiles/speechcommand_valid_data.json --data-eval ./data/datafiles/speechcommand_eval_data.json --exp-dir ./exp/test-speechcommands-f10-t10-pTrue-b32-lr2.5e-4-decoupe --label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 --lr 2.5e-4 --n-epochs 2 --batch-size 16 --save_model True --freqm 48 --timem 48 --mixup 0.6 --bal none --dataset_mean -6.845978 --dataset_std 5.5654526 --audio_length 128 --noise True --metrics acc --loss BCE --warmup False --lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --tstride 10 --fstride 10 --imagenet_pretrain True --audioset_pretrain False
    ```
    ```

    Command for no-kd

    CUDA_VISIBLE_DEVICES=1 python -W ignore ../../src/run_kd.py --model ast --dataset audioset --data-train ./data/datafiles/balanced_train_segments.json --data-val ./data/datafiles/eval_segments.json --exp-dir ./exp_non_kd/ --label-csv ./data/class_labels_indices.csv --n_class 527 --lr 2.5e-4 --n-epochs 16 --batch-size 8 --save_model True --freqm 48 --timem 192 --mixup 0.5 --bal none --tstride 10 --fstride 12 --imagenet_pretrain True --dataset_mean -4.2677393 --dataset_std 4.5689974 --audio_length 1024 --noise False --metrics mAP --loss BCE --warmup True --lrscheduler_start 10 --lrscheduler_step 5 --lrscheduler_decay 0.5 --wa True --wa_start 6 --wa_end 25 --kd_lambda 1.0

    ```

    Command for kd-version
    CUDA_VISIBLE_DEVICES=2 python -W ignore ../../src/run_kd.py --model ast --dataset audioset --data-train ./data/datafiles/balanced_train_segments.json --data-val ./data/datafiles/eval_segments.json --exp-dir ./exp_kd/ --label-csv ./data/class_labels_indices.csv --n_class 527 --lr 2.5e-4 --n-epochs 16 --batch-size 8 --save_model True --freqm 48 --timem 192 --mixup 0.5 --bal none --tstride 10 --fstride 12 --imagenet_pretrain True --dataset_mean -4.2677393 --dataset_std 4.5689974 --audio_length 1024 --noise False --metrics mAP --loss BCE --warmup True --lrscheduler_start 10 --lrscheduler_step 5 --lrscheduler_decay 0.5 --wa True --wa_start 6 --wa_end 25 --kd_lambda 0.1
    
    ```



## Inference for Speech Command (Evaluation on Speechcommand dataset):

4. Navigate to the project directory:
    ```bash
    cd /data/swarup_behera/Research/TOME/ToMe
    ```

5. Run the evaluation script:
    ```bash
    python evaluation_tome.py
    ```

## Details of TOME or Patching AST Model (Our Contribution):

In the `src/run.py` file 132 line number, the following lines were added to patch the AST model with TOME, along with modifications in TOME files:

```python
import sys

sys.path.append('../../')
from evaluation import apply_patch
apply_patch(audio_model)
audio_model.r = 32
sys.path.append(basepath)


We did experiment with different tome numbers.


Next todo:

1) (Friday)Youtube dl, issue for downloading audioset, download 20k samples.
2) (saturday or Monday)  we will run overnight 20 epochs on differnet value of r(0 tp 64).
3) (For next week)  we will write kd loss for fasterast.
4) Same above experiment for kd+faster-ast, and report the accuracy and losses.






Check the exact numbers:

Training Audios: 9307
Validation Audios: 9980

27 Jan 2024:


1) Increase the training data.(Later)
2) we will add another metric(mAP).(today)
3) We will add logging.(today)
4) kd vs non-kd loss.(today)[Experiment are done with r=0]
5) We should use cnn logists.

4) Experiment with different values for R for tome improvements. (todo)
5) (karthik) Run the ast-model and compare the parameters in different ast models in timm library(done but dission pending).


error with file:
Debug for below file:
/data/swarup_behera/Research/TOME/ToMe/egs/audioset/audios/eval_segments/YGAohd8KvONo.wav




29 Jan 2024:

1) Running no-tome for kd vs non-kd, and compare mAP, Accuracy and dump the logs.(today)
2) if above experimet given better result for kd.(today)




30 Jan task:
0) Compare kd vs non-kd results for r=0, and passt logists.(tomorrow)
1) run below inference on one audio and get result.(done)
https://github.com/fschmid56/EfficientAT/blob/main/inference.py


2) Bench-marking results:

Run inference on all the audios we have in training, and calculate accuracy, and mAP.(Tomorrow with origin codebase).(bottleneckl)(Pending till problem with nvidia-smi).


3) Get the logists for all the audios(done)



Activities for this week ( thursday and Friday):

1) Download pretrain CNN model on audioset.(done)
2) Generate logits for the wav files we have(527 values).(done)
3) Use above logits in Kd for ast.(todo)