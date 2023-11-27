# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
import os, csv, argparse
import torch, torchaudio, timm
import time
from torch import nn
import numpy as np
from torch.cuda.amp import autocast
from typing import Tuple
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r
from src.models import ASTModel
from src import  dataloader
import torch.nn.functional as F
from src.utilities import *
class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # import pdb;pdb.set_trace()       
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        # print(len(self._tome_info["r"]))
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # import pdb;pdb.set_trace()
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        # if size is not None:
        #     attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.v.blocks)+168, self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels
def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.v.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True
    # print(model.v.modules))
    # for module in model.modules():
    #     if isinstance(module, Block):
    #         module.__class__ = ToMeBlock
    #         module._tome_info = model._tome_info
    #     elif isinstance(module, Attention):
    #         module.__class__ = ToMeAttention
    for count, module in enumerate(model.modules(), start=1):
        # if 'Mlp' in str(module.__class__):
        #     c+=1
        # if(count>77):
        #     break
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
def validate(audio_model, val_loader, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = time.time()
    # if not isinstance(audio_model, nn.DataParallel):
    #     audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    import tqdm
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(tqdm.tqdm(val_loader)):
            audio_input = audio_input.to(device)
            labels = labels.to(device)
            
            # compute output
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output).to(device)
            
            loss = nn.BCEWithLogitsLoss()(audio_output, labels.float())  # Use labels directly
            
            predictions = audio_output.to('cpu').detach()
            
            A_predictions.append(predictions)
            A_targets.append(labels)
            A_loss.append(loss.to('cpu').detach())
        A_loss.append(loss.to('cpu').detach())
        batch_time += time.time() - end
        print("REsult ")
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        audio_output = audio_output.cpu().numpy()
        target = target.cpu().numpy()
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        print(batch_time)
        exp_dir = "exp"
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss
'''
# Create an AST model and download the AudioSet pretrained weights
# audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
# if os.path.exists('/data/swarup_behera/Research/TOME/ToMe/pretrained_models/audio_mdl.pth') == False:
#     wget.download(audioset_mdl_url, out='/data/swarup_behera/Research/TOME/ToMe/pretrained_models/audio_mdl.pth')

# Assume each input spectrogram has 1024 time frames
input_tdim = 1024
checkpoint_path = '/data/swarup_behera/Research/TOME/ToMe/pretrained_models/audioset_0.4593.pth?dl=1'
# now load the visualization model
# ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
# input_tdim = 100
# now load the visualization model
ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
print(f'[*INFO] load checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cuda')
audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
audio_model.load_state_dict(checkpoint)
audio_model = audio_model.to(torch.device("cuda:0"))
audio_model.eval()   



input_tdim = 1024 
feats = make_features('sample_audio/sample_audio.flac', mel_bins=128)           # shape(1024, 128)
feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature
feats_data = feats_data.to(torch.device("cuda:0"))


# input_tdim = 100  # Replace with your desired size

# feats_data = feats.unsqueeze(0).expand(input_tdim, feats.size(0), feats.size(1))
# feats_data = feats_data.to(torch.device("cuda:0"))


# Make the prediction
with torch.no_grad():
  with autocast():
    output = ast_mdl.forward(feats_data)
    output = torch.sigmoid(output)
result_output = output.data.cpu().numpy()[0]
sorted_indexes = np.argsort(result_output)[::-1]
label_csv = '/data/swarup_behera/Research/TOME/ast/egs/audioset/data/class_labels_indices.csv'       # label and indices for audioset data
labels = load_label(label_csv)
# Print audio tagging top probabilities
print('Predice results:')
for k in range(10):
    print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))


# apply_patch(ast_mdl)
# ToMe with r=16
# ast_mdl.r = 16

output = ast_mdl.forward(feats_data)
output = torch.sigmoid(output)
result_output = output.data.cpu().numpy()[0]
sorted_indexes = np.argsort(result_output)[::-1]
# Print audio tagging top probabilities
print('Predice results:')
for k in range(10):
    print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))




# if args.dataset == 'speechcommands':
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
# audio_model = torch.nn.DataParallel(audio_model)
# audio_model.load_state_dict(sd)

# best model on the validation set
# note it is NOT mean of class-wise accuracy

# for speechcommands dataset, evaluate the best model on validation set on the test set


print('---------------evaluate on the validation set---------------')

data_eval = "/data/swarup_behera/Research/TOME/ToMe/egs/speechcommands/data/datafiles/speechcommand_eval_data.json"
label_csv = '/data/swarup_behera/Research/TOME/ast/egs/speechcommands/data/speechcommands_class_labels_indices.csv'       # label and indices for audioset data

val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': "speechcommands", 'mode':'evaluation', 'mean': -6.845978 , 'std':5.5654526, 'noise':False}
batch_size = 2
# test the model on the evaluation set
eval_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(data_eval, label_csv=label_csv, audio_conf=val_audio_conf),
    batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)


ast_mdl = ASTModel(label_dim=35, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
print(f'[*INFO] load checkpoint: {checkpoint_path}')

checkpoint = torch.load(checkpoint_path, map_location='cuda')
audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
# audio_model.load_state_dict(checkpoint)
audio_model = audio_model.to(torch.device("cuda:0"))
apply_patch(ast_mdl)
# ToMe with r=16
ast_mdl.r = 16
audio_model.eval()  

stats, _ = validate(ast_mdl, eval_loader, 'eval_set')
eval_acc = stats[0]['acc']
eval_mAUC = np.mean([stat['auc'] for stat in stats])
print('---------------evaluate on the test set---------------')
print("Accuracy: {:.6f}".format(eval_acc))
print("AUC: {:.6f}".format(eval_mAUC))
exp_dir = "exp"
np.savetxt(exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
'''