# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to train Diffusion Policy on the PushT environment."""

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

def main():

    #Log to Tesnorboard
    data_name = "my_pusht"
    writer = SummaryWriter('runs/{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), data_name))

    # Create a directory to store the training checkpoint.
    base_output_dir = f"outputs/train/my_pusht_diffusion/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_directory = Path(base_output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 150000
    save_interval = 25000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    pusht_dir = f"lerobot/{data_name}"
    data_dir = os.environ['HOME'] + f"/training_data/{pusht_dir}"

    dataset_metadata = LeRobotDatasetMetadata(data_dir)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon = 16,
        n_obs_steps = 2,
        n_action_steps = 8,
        drop_n_last_frames = 7,
        vision_backbone = "resnet18",
        crop_shape = (184, 224),
        crop_is_random = True,  
        pretrained_backbone_weights = None,
        use_group_norm = True,
        spatial_softmax_num_keypoints = 24, 
        use_separate_rgb_encoder_per_camera = False, 
        down_dims = (64, 128, 256),
        kernel_size = 5,
        n_groups = 8,
        diffusion_step_embed_dim = 128,
        use_film_scale_modulation = True,
        noise_scheduler_type = "DDPM",
        num_train_timesteps = 100,
        beta_schedule = "squaredcos_cap_v2", 
        beta_start = 0.0001,
        beta_end = 0.02,
        prediction_type = "epsilon",
        clip_sample = True,
        clip_sample_range = 1.0,
        num_inference_steps = None,
        do_mask_loss_for_padding = False,
        optimizer_lr = 1e-4,
        optimizer_betas = (0.95, 0.999),
        optimizer_eps = 1e-8,
        optimizer_weight_decay = 0.0,
        scheduler_name = "cosine",
        scheduler_warmup_steps = 500, 
        use_amp = True
        )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(data_dir, delta_timestamps=delta_timestamps)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=cfg.optimizer_lr,
        betas = cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay
        )

    num_workers = 4
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step}/{training_steps} loss: {loss.item():.4f}")
                writer.add_scalar('loss', loss, step)
            step += 1
            if step % save_interval == 0:
                print('\033[32m'+"Saving model!"+'\033[0m')
                # Save a policy checkpoint.
                model_save_dir = base_output_dir + f"/{step}"
                mpde_save_path = Path(model_save_dir)
                mpde_save_path.mkdir(parents=True, exist_ok=True)

                policy.save_pretrained(mpde_save_path)
                preprocessor.save_pretrained(mpde_save_path)
                postprocessor.save_pretrained(mpde_save_path)

            if step >= training_steps+1:
                done = True
                break

if __name__ == "__main__":
    main()
