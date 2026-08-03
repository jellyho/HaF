"""Config mirroring src/haf/models/haf_config.py (HAFConfig) for the torch/SmolVLM stack."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HAFTorchConfig:
    # backbone
    vlm_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    dtype: str = "bfloat16"
    freeze_vlm: bool = False
    # SmolVLM/Idefics3 tiles each image (~17 tiles => ~1139 tokens => OOM). One robot view is enough.
    image_splitting: bool = False
    vlm_image_size: int = 384              # tile longest edge fed to the VLM                    # False = fine-tune the pretrained VLM (the real-VLA regime)

    # action space (RT-1/fractal defaults)
    action_dim: int = 7
    action_horizon: int = 15

    # action expert (flow matching), mirrors pi0 action expert
    expert_width: int = 512
    expert_depth: int = 2
    flow_steps_train: int = 1                   # single random t per sample
    flow_steps_sample: int = 10
    flow_samples: int = 8

    # ---- objectives (mirrors HAFConfig enable_* + *_loss_weight) ----
    enable_action_training: bool = True
    action_loss_weight: float = 1.0

    # AHA auxiliary (mirrors retro_embedding_loss_weight / stop_retro_to_vlm_grad)
    aux_loss_weight: float = 0.0                # 0 = inert (BC-only baseline)
    stop_aux_to_vlm_grad: bool = False          # True = KI control (aux does NOT shape the VLM)
    stop_bc_to_vlm_grad: bool = False           # True = insulate the VLM from the BC head (pi0.5-style KI)

    # artificial objective family (the confound-free recoverability dial)
    #   'mask'  : structured-hard   — predict clean joint rep from a MASKED image forward (ratio dials difficulty)
    #   'noise' : noise-hard        — predict a target corrupted with gaussian noise (sigma dials difficulty)
    #   'future': natural           — predict a frozen embedding of a future frame (offset dials difficulty)
    aux_family: str = "mask"
    aux_mask_ratio: float = 0.5                 # for family='mask'
    aux_noise_sigma: float = 0.0                # for family='noise'
    aux_future_offset: int = 5                  # for family='future'

    # optimization
    lr: float = 1e-5
    weight_decay: float = 1e-2
    batch_size: int = 8
    grad_clip: float = 1.0
    max_steps: int = 20000
    seed: int = 0

    # data
    image_size: int = 224
    prompt_prefix: str = ""                     # optional instruction prefix

    def describe(self) -> str:
        aux = "off" if self.aux_loss_weight == 0 else (
            f"{self.aux_family}:" + {
                "mask": f"ratio={self.aux_mask_ratio}",
                "noise": f"sigma={self.aux_noise_sigma}",
                "future": f"offset={self.aux_future_offset}",
            }[self.aux_family] + f" w={self.aux_loss_weight}"
            + (" [KI: aux insulated]" if self.stop_aux_to_vlm_grad else "")
        )
        return (f"SmolVLM-VLA | vlm={self.vlm_id.split('/')[-1]} frozen={self.freeze_vlm} | "
                f"action {self.action_horizon}x{self.action_dim} flow | aux {aux} | "
                f"bc_ki={self.stop_bc_to_vlm_grad} | lr={self.lr} bs={self.batch_size}")
