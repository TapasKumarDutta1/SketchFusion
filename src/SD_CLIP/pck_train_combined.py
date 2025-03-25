import argparse
import os
import sys

import torch
from itertools import chain
from loguru import logger
from typing import *

from PIL import Image
from PIL import *  

from torch import nn
from torch.nn import Conv2d
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np

from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel


from tqdm import tqdm
import yaml

# Custom imports
from model_utils.projection_network import AggregationNetwork
from model_utils.corr_map_model import Correlation2Displacement
import utils.utils_losses as utils_losses

from utils.utils_dataset import load_and_prepare_data, load_eval_data
from utils.utils_correspondence import kpts_to_patch_idx, load_img_and_kps, convert_to_binary_mask, calculate_keypoint_transformation, get_distance, get_distance_mutual_nn
sys.path.append("./sketchfusion/src/CLIP")
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"

def resize(img, target_res=224, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class OneStepSDPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        img_tensor,
        vit_model,
        clip_sample,
        t,
        up_ft_indices,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        device = self._execution_device
        latents = (
            self.vae.encode(img_tensor).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy,
            vit_model,
            clip_sample,
            t,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return unet_output

class MyUNet2DConditionModel(UNet2DConditionModel):

    def forward(
        self,
        sample: torch.FloatTensor,
        vit_model: torch.FloatTensor,
        clip_sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        clip_x1 = clip_sample

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        count_number = 0
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

            if count_number == 0:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_1(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if count_number == 1:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_2(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if count_number == 2:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_3(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if count_number == 3:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_4(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

            sample += clip_x2
            count_number += 1

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            if count_number == 4:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_5(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if count_number == 5:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_6(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if count_number == 6:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_7(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
            if count_number == 7:
                clip_x2 = F.interpolate(
                    vit_model.train_conv_8(clip_x1),
                    size=(
                        vit_model.spatial_size[count_number],
                        vit_model.spatial_size[count_number],
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

            sample += clip_x2
            count_number += 1
            if i <= np.max(up_ft_indices):
                up_ft[i] = sample

        for i in up_ft.keys():
            up_ft[i] = up_ft[i].detach()
        output = {}
        output["up_ft"] = up_ft
        return output


class SDFeaturizer:
    def __init__(self, sd_id="stabilityai/stable-diffusion-2-1-base"):
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(
            sd_id, unet=unet, safety_checker=None
        )
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        null_prompt = ""
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )  

        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt
        self.pipe = onestep_pipe

    def forward(
        self,
        img_tensor,  
        clip,
        vit_features,
        prompt="",
        t=195,
        ensemble_size=32,
    ):
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() 

        prompt_embeds = self.pipe._encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )  
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all_2 = self.pipe(
            img_tensor=img_tensor,
            vit_model=clip,
            clip_sample=vit_features,
            t=t,
            up_ft_indices=[2],
            prompt_embeds=prompt_embeds,
        )
        return {
            "s5": unet_ft_all_2["up_ft"][0].mean(0).unsqueeze(0),
            "s4": unet_ft_all_2["up_ft"][1].mean(0).unsqueeze(0),
            "s3": unet_ft_all_2["up_ft"][2].mean(0).unsqueeze(0),
        }


def normalize_feats(args, feats, epsilon=1e-10):
    norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
    norm_feats = feats / (norms + epsilon)

    return norm_feats


def process_and_save_features(
    file_path,
    model,
    extractor_vit,
    extracted_features,
    img_size,
):
    img1 = Image.open(file_path).convert("RGB")
    img1 = resize(img1, img_size, resize=True, to_pil=True)
    sd_image = torch.from_numpy(
        np.transpose(np.expand_dims(np.array(img1), 0), (0, 3, 1, 2))
    )
    # img1_batch = extractor_vit.preprocess_pil(img1)
    accumulated_features = model.forward(
        sd_image.float().to("cuda"), extractor_vit, extracted_features, ""
    )
    return accumulated_features


def prepare_feature_paths_and_load(
    aggre_net,
    img_path,
    num_patches,
    device,
    mod,
    model,
    preprocess,
    img_size,
):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    _, image_features_clip = model(image)
    desc_clip = image_features_clip[1:, :, :].permute(1, 2, 0).reshape(-1, 1024, 24, 24)

    features_sd = process_and_save_features(
        img_path, mod, model, desc_clip, img_size
    )
    for k in features_sd:
        features_sd[k] = features_sd[k].to(device)
    desc_gathered = torch.cat(
        [
            F.interpolate(
                features_sd["s3"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_sd["s4"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                features_sd["s5"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
        ],
        dim=1,
    )
    desc = aggre_net(desc_gathered)
    desc = desc.reshape(1, 1, -1, num_patches**2).permute(0, 1, 3, 2)
    return desc, None


def get_patch_descriptors(
    args,
    aggre_net,
    num_patches,
    files,
    pair_idx,
    mod,
    model,
    preprocess,
    img_size,
    device="cuda",
):
    img_path_1 = files[pair_idx * 2]
    img_path_2 = files[pair_idx * 2 + 1]
    # save the imgs for cases if the feature doesn't exist
    img1_desc, mask1 = prepare_feature_paths_and_load(
        aggre_net,
        img_path_1,
        num_patches,
        device,
        mod,
        model,
        preprocess,
        img_size,
    )
    img2_desc, mask2 = prepare_feature_paths_and_load(
        aggre_net,
        img_path_2,
        num_patches,
        device,
        mod,
        model,
        preprocess,
        img_size,
    )
    # normalize the desc
    img1_desc = normalize_feats(args, img1_desc[0])
    img2_desc = normalize_feats(args, img2_desc[0])
    return img1_desc, img2_desc, mask1, mask2


def train(
    args,
    aggre_net,
    corr_map_net,
    optimizer,
    scheduler,
    logger,
    mod,
    model,
    preprocess,
):
    # gather training data
    files, kps, _, all_thresholds = load_and_prepare_data(args)
    # train
    num_patches = args.NUM_PATCHES
    N = len(files) // 2
    pbar = tqdm(total=N)

    img_size, loss_count, count = 336,  0, 0


    for epoch in range(args.EPOCH):
        pbar.reset()
        for j in range(0, N, args.BZ):
            optimizer.zero_grad()
            batch_loss = 0  # collect the loss for each batch
            for pair_idx in range(j, min(j + args.BZ, N)):
                # Load images and keypoints
                img1_kps = kps[2 * pair_idx]
                img2_kps = kps[2 * pair_idx + 1]
                # Get patch descriptors/feature maps
                img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(
                    args,
                    aggre_net,
                    num_patches,
                    files,
                    pair_idx,
                    mod,
                    model,
                    preprocess,
                    img_size,
                )
                args.ADAPT_FLIP, args.AUGMENT_SELF_FLIP, args.AUGMENT_DOUBLE_FLIP = (
                    0,
                    0,
                    0,
                )  # augment with flip)
                img1_desc_flip = img2_desc_flip = raw_permute_list = None
                # img1_desc_flip = img2_desc_flip = raw_permute_list = None
                # Get the threshold for each patch
                scale_factor = num_patches / args.ANNO_SIZE
                if args.BBOX_THRE:
                    img1_threshold = all_thresholds[2 * pair_idx] * scale_factor
                    img2_threshold = all_thresholds[2 * pair_idx + 1] * scale_factor
                else:  # image threshold
                    img1_threshold = img2_threshold = args.ANNO_SIZE

                # Compute loss
                loss = utils_losses.calculate_loss(
                    args,
                    aggre_net,
                    img1_kps,
                    img2_kps,
                    img1_desc,
                    img2_desc,
                    img1_threshold,
                    img2_threshold,
                    mask1,
                    mask2,
                    num_patches,
                    device,
                    raw_permute_list,
                    img1_desc_flip,
                    img2_desc_flip,
                    corr_map_net,
                )

                # Accumulate loss over iterations
                loss_count += loss.item()
                count += 1
                batch_loss += loss
                pbar.update(1)

                with torch.no_grad():
                    # Log loss periodically or at the end of the dataset
                    if (
                        pair_idx % 100 == 0 and pair_idx > 0
                    ) or pair_idx == N - 1:  # Log every 100 iterations and at the end of the dataset
                        logger.info(
                            f"Step {pair_idx + epoch * N} | Loss: {loss_count / count:.4f}"
                        )
                        loss_count = count = 0  # reset loss count
                    # Evaluate model periodically, at the end of the dataset, or under specific conditions

            batch_loss /= args.BZ
            batch_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        print("EPOCH: " + str(epoch) + " DONE")
        
        eval(args, aggre_net, mod, model, preprocess)



class CLIP_Image(nn.Module):
    def __init__(self, model):
        super(CLIP_Image, self).__init__()
        self.model = model
        self.train_conv_1 = Conv2d(1024, 320, kernel_size=(1, 1), stride=(1, 1))
        self.train_conv_2 = Conv2d(1024, 640, kernel_size=(1, 1), stride=(1, 1))
        self.train_conv_3 = Conv2d(1024, 1280, kernel_size=(1, 1), stride=(1, 1))
        self.train_conv_4 = Conv2d(1024, 1280, kernel_size=(1, 1), stride=(1, 1))
        self.train_conv_5 = Conv2d(1024, 1280, kernel_size=(1, 1), stride=(1, 1))
        self.train_conv_6 = Conv2d(1024, 1280, kernel_size=(1, 1), stride=(1, 1))
        self.train_conv_7 = Conv2d(1024, 640, kernel_size=(1, 1), stride=(1, 1))
        self.spatial_size = [21, 11, 6, 6, 11, 21, 42]

    def forward(self, x):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.model.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        feature_maps = self.model.transformer(x)
        x = feature_maps.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_post(x[:, 0, :])
        if self.model.proj is not None:
            x = x @ self.model.proj

        return x, feature_maps


def main(args):
    set_seed(args.SEED)
    args.NUM_PATCHES = 60
    args.BBOX_THRE = not (args.IMG_THRESHOLD or args.EVAL_DATASET == "pascal")
    args.AUGMENT_FLIP, args.AUGMENT_DOUBLE_FLIP, args.AUGMENT_SELF_FLIP = (
        (1.0, 1.0, 0.25) if args.PAIR_AUGMENT else (0, 0, 0)
    )  # set different weight for different augmentation
    feature_dims = [
        640,
        1280,
        1280,
    ]  # dimensions for three layers of SD and one layer of CLIP features

    # Determine the evaluation type and project name based on args
    save_path = f"./results_{args.EVAL_DATASET}/pck_train_{args.NOTE}_sample_{args.EPOCH}_{args.SAMPLE}_lr_{args.LR}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = get_logger(save_path + "/result.log")
    logger.info(args)
    aggre_net = AggregationNetwork(
        feature_dims=feature_dims,
        projection_dim=args.PROJ_DIM,
        device=device,
        feat_map_dropout=args.FEAT_MAP_DROPOUT,
    )
    aggre_net.to(device)
    total_args = aggre_net.parameters()
    if args.DENSE_OBJ > 0:
        corr_map_net = Correlation2Displacement(
            setting=args.DENSE_OBJ, window_size=args.SOFT_TRAIN_WINDOW
        ).to(device)
        total_args = chain(total_args, corr_map_net.parameters())
    else:
        corr_map_net = None

    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model = clip_model.float()
    model = CLIP_Image(clip_model.visual).float()
    model.to(device)
    for name, param in model.named_parameters():
        if "train_" not in name:
            param.requires_grad = False
        else:
            print(name)

    optimizer = torch.optim.AdamW(
        chain(total_args, model.parameters()), lr=args.LR, weight_decay=args.WD
    )
    if args.SCHEDULER is not None:
        if args.SCHEDULER == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=5000 * args.EPOCH, eta_min=1e-6
            )  # 53339 is the number of training pairs for SPair-71k
        if args.SCHEDULER == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.LR,
                steps_per_epoch=32,
                epochs=args.EPOCH,
                pct_start=args.SCHEDULER_P1,
            )
    else:
        scheduler = None

    mod = SDFeaturizer()
    train(
            args,
            aggre_net,
            corr_map_net,
            optimizer,
            scheduler,
            logger,
            mod,
            model,
            preprocess,
        )


def compute_pck(
    args,
    aggre_net,
    files,
    kps,
    mod,
    model,
    preprocess,
    thresholds=None,
):
    out_results = []
    num_patches = args.NUM_PATCHES
    (
        gt_correspondences,
        pred_correspondences,
        img_acc_001,
        img_acc_005,
        img_acc_01,
        len_kpts,
    ) = ([] for _ in range(6))
    N = len(files) // 2
    pbar = tqdm(total=N, position=0, leave=True)

    error_average = []
    img_size = 336

    for pair_idx in range(N):
        # Load images and keypoints
        img1, img1_kps = load_img_and_kps(
            idx=2 * pair_idx, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False
        )
        img2, img2_kps = load_img_and_kps(
            idx=2 * pair_idx + 1,
            files=files,
            kps=kps,
            img_size=args.ANNO_SIZE,
            edge=False,
        )
        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        vis2 = img2_kps[:, 2]
        # Get patch descriptors
        with torch.no_grad():
            img1_desc, img2_desc, _, _ = get_patch_descriptors(
                args,
                aggre_net,
                num_patches,
                files,
                pair_idx,
                mod,
                model,
                preprocess,
                img_size
            )
        # Get patch index for the keypoints
        img1_patch_idx = kpts_to_patch_idx(args, img1_kps, num_patches)
        # Get similarity matrix
        kps_1_to_2 = calculate_keypoint_transformation(
            args, img1_desc, img2_desc, img1_patch_idx, num_patches
        )

        # collect the result for more complicated eval
        single_result = {
            "src_fn": files[2 * pair_idx],  # must
            "trg_fn": files[2 * pair_idx + 1],  # must
            "resize_resolution": args.ANNO_SIZE,  # must
        }
        out_results.append(single_result)

        gt_kps = img2_kps[vis][:, [1, 0]]
        prd_kps = kps_1_to_2[vis][:, [1, 0]]
        gt_correspondences.append(gt_kps)
        pred_correspondences.append(prd_kps)
        len_kpts.append(vis.sum().item())

        # compute per image acc
        if not args.KPT_RESULT:  # per img result
            single_gt_correspondences = img2_kps[vis][:, [1, 0]]
            single_pred_correspondences = kps_1_to_2[vis][:, [1, 0]]
            alpha = (
                torch.tensor([0.1, 0.05, 0.01])
                if args.EVAL_DATASET != "pascal"
                else torch.tensor([0.1, 0.05, 0.15])
            )
            correct = torch.zeros(3)
            err = (single_gt_correspondences - single_pred_correspondences.cpu()).norm(
                dim=-1
            )
            error_average.append(err)
            err = err.unsqueeze(0).repeat(3, 1)
            if thresholds is not None:
                single_bbox_size = (
                    torch.tensor(thresholds[pair_idx]).repeat(vis.sum()).cpu()
                )
                correct += (
                    (err < alpha.unsqueeze(-1) * single_bbox_size.unsqueeze(0))
                    .float()
                    .mean(dim=-1)
                )
            else:
                correct += (
                    (err < alpha.unsqueeze(-1) * args.ANNO_SIZE).float().mean(dim=-1)
                )
            img_acc_01.append(correct[0].item())
            img_acc_005.append(correct[1].item())
            img_acc_001.append(correct[2].item())

        pbar.update(1)
    print("PCK10: " + str(np.mean(img_acc_01)) + "PCK05: " + str(np.mean(img_acc_005)))
    return (
        np.mean(img_acc_01),
        np.mean(img_acc_005),
        torch.stack(error_average).mean().numpy(),
    )


def eval(args, aggre_net, mod, model, preprocess, split="val"):
    aggre_net.eval()  # Set the network to evaluation mode
    # Configure data directory and categories based on the dataset type


    # Process each category
    files, kps, thresholds, _ = load_eval_data(args)
    p10, p05, error = compute_pck(
        args,
        aggre_net,
        files,
        kps,
        mod,
        model,
        preprocess,
        thresholds=thresholds,
    )
    return p10, p05, error
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # load config
    parser.add_argument("--config", type=str, default=None)  # path to the config file

    # basic training setting
    parser.add_argument("--SEED", type=int, default=42)  # random seed
    parser.add_argument("--NOTE", type=str, default="")  # note for the experiment
    parser.add_argument(
        "--SAMPLE", type=int, default=0
    )  # sample 100 pairs for each category for training, set to 0 to use all pairs
    parser.add_argument(
        "--TEST_SAMPLE", type=int, default=20
    )  # sample 20 pairs for each category for testing, set to 0 to use all pairs
    parser.add_argument(
        "--TOTAL_SAVE_RESULT", type=int, default=0
    )  # save the qualitative results for the first 5 pairs
    parser.add_argument(
        "--IMG_THRESHOLD", action="store_true", default=False
    )  # set the pck threshold to the image size rather than the bbox size
    parser.add_argument(
        "--ANNO_SIZE", type=int, default=336
    )  # image size for the annotation input
    parser.add_argument("--LR", type=float, default=1.25e-3)  # learning rate
    parser.add_argument("--WD", type=float, default=1e-3)  # weight decay
    parser.add_argument("--BZ", type=int, default=1)  # batch size
    parser.add_argument(
        "--SCHEDULER", type=str, default=None
    )  # set to use lr scheduler, one_cycle, cosine, plateau
    parser.add_argument(
        "--SCHEDULER_P1", type=float, default=0.3
    )  # set the first parameter for the scheduler
    parser.add_argument("--EPOCH", type=int, default=10)  # number of epochs
    parser.add_argument(
        "--EVAL_EPOCH", type=int, default=1
    )  # number of steps for evaluation
    parser.add_argument(
        "--NOT_WANDB", action="store_true", default=False
    )  # set true to not use wandb
    parser.add_argument(
        "--TRAIN_DATASET", type=str, default="spair"
    ) 

    # training model setup
    parser.add_argument(
        "--LOAD", type=str, default=None
    )  # path to load the pretrained model
    parser.add_argument(
        "--DENSE_OBJ", type=int, default=1
    )  # set true to use the dense training objective, 1: enable; 0: disable
    parser.add_argument(
        "--GAUSSIAN_AUGMENT", type=float, default=0.1
    )  # set float to use the gaussian augment, float for std
    parser.add_argument(
        "--FEAT_MAP_DROPOUT", type=float, default=0.5
    )  # set true to use the dropout for the feat map
    parser.add_argument(
        "--ENSEMBLE", type=int, default=32
    )  # set true to use the ensembles of sd feature maps
    parser.add_argument(
        "--PROJ_DIM", type=int, default=1280
    )  # projection dimension of the post-processor
    parser.add_argument(
        "--PAIR_AUGMENT", action="store_true", default=False
    )  # set true to enable pose-aware pair augmentation
    parser.add_argument(
        "--SELF_CONTRAST_WEIGHT", type=float, default=0
    )  # set true to use the self supervised loss
    parser.add_argument(
        "--SOFT_TRAIN_WINDOW", type=int, default=0
    )  # set true to use the window soft argmax during training, default is using standard soft argmax

    # evaluation setup
    parser.add_argument(
        "--DO_EVAL", action="store_true", default=False
    )  # set true to do the evaluation on test set
    parser.add_argument(
        "--DUMMY_NET", action="store_true", default=True
    )  # set true to use the dummy net, used for zero-shot setting
    parser.add_argument(
        "--EVAL_DATASET", type=str, default="spair"
    )  # set the evaluation dataset, 'spair' for SPair-71k, 'pascal' for PF-Pascal, 'ap10k' for AP10k
    parser.add_argument(
        "--AP10K_EVAL_SUBSET", type=str, default="intra-species"
    )  # set the test setting for ap10k dataset, `intra-species`, `cross-species`, `cross-family`
    parser.add_argument(
        "--KPT_RESULT", action="store_true", default=False
    )  # set true to evaluate per kpt result, in the paper, this is used for comparing unsupervised methods, following ASIC
    parser.add_argument(
        "--ADAPT_FLIP", action="store_true", default=False
    )  # set true to use the flipped images, adaptive flip
    parser.add_argument(
        "--MUTUAL_NN", action="store_true", default=False
    )  # set true to use the flipped images, adaptive flip, mutual nn as metric
    parser.add_argument(
        "--SOFT_EVAL", action="store_true", default=False
    )  # set true to use the soft argmax eval
    parser.add_argument(
        "--SOFT_EVAL_WINDOW", type=int, default=7
    )  # set true to use the window soft argmax eval, window size is 2*SOFT_EVAL_WINDOW+1, 0 to be standard soft argmax

    args = parser.parse_args()
    if args.config is not None:  # load config file and update the args
        args_dict = vars(args)
        args_dict.update(load_config(args.config))
        args = argparse.Namespace(**args_dict)
    main(args)