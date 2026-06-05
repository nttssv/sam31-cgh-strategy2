#!/usr/bin/env python3
"""Patch the official SAM3 checkout for this V100/Jupyter training setup."""

from __future__ import annotations

import argparse
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def patch_vitdet(sam3_repo: Path) -> None:
    path = sam3_repo / "sam3" / "model" / "vitdet.py"
    text = path.read_text()
    old = """    def forward(self, x):
        x = addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
"""
    new = """    def forward(self, x):
        if torch.is_grad_enabled():
            x = self.fc1(x)
            x = self.act(x)
        else:
            x = addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
"""
    if new in text:
        print("vitdet.py already patched")
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected Mlp.forward block in {path}")
    path.write_text(text.replace(old, new))
    print("patched", path)


def patch_model_builder(sam3_repo: Path) -> None:
    path = sam3_repo / "sam3" / "model_builder.py"
    text = path.read_text()
    old = """    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model
"""
    new = """    # Freeze heavy pretrained backbones by default for small-GPU fine-tuning.
    if not eval_mode and os.environ.get("SAM3_CGH_FREEZE_BACKBONES", "1") != "0":
        backbone = getattr(model, "backbone", None)
        for module_name in ("vision_backbone", "language_backbone"):
            module = getattr(backbone, module_name, None) if backbone is not None else None
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model
"""
    if new in text:
        print("model_builder.py already patched")
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected build_sam3_image_model return block in {path}")
    path.write_text(text.replace(old, new, 1))
    print("patched", path)


def patch_optimizer(sam3_repo: Path) -> None:
    path = sam3_repo / "sam3" / "train" / "optim" / "optimizer.py"
    text = path.read_text()
    replacements = {
        """    model_parameters = {parameter for _, parameter in model.named_parameters()}
""": """    model_parameters = {
        parameter for _, parameter in model.named_parameters() if parameter.requires_grad
    }
""",
        """    if param_allowlist is None:
        param_allowlist = {name for name, _ in model.named_parameters()}
""": """    if param_allowlist is None:
        param_allowlist = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
""",
        """        matching_parameters = set(fnmatch.filter(parameter_names, param_name))
        assert len(matching_parameters) >= 1, (
            f"param_name {param_name} does not match any parameters in the model"
        )
        logging.info(f"Matches for param_name [{param_name}]: {matching_parameters}")
        allowed_parameter_names.append(matching_parameters)
    return set.union(*allowed_parameter_names)
""": """        matching_parameters = set(fnmatch.filter(parameter_names, param_name))
        if len(matching_parameters) == 0:
            logging.info(
                f"Skipping optimizer pattern [{param_name}] because it matches no trainable parameters"
            )
            continue
        logging.info(f"Matches for param_name [{param_name}]: {matching_parameters}")
        allowed_parameter_names.append(matching_parameters)
    return set.union(*allowed_parameter_names) if allowed_parameter_names else set()
""",
    }
    changed = False
    for old, new in replacements.items():
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"Could not find expected optimizer block in {path}")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print("patched", path)
    else:
        print("optimizer.py already patched")


def patch_logger(sam3_repo: Path) -> None:
    path = sam3_repo / "sam3" / "train" / "utils" / "logger.py"
    text = path.read_text()
    replacements = {
        """import functools
import logging
import sys
import uuid
""": """import functools
import logging
import os
import sys
import uuid
""",
        """class Logger:
    \"\"\"
    A logger class that can interface with multiple loggers. It now supports tensorboard only for simplicity, but you can extend it with your own logger.
    \"\"\"
""": """class WandBLogger:
    def __init__(
        self,
        project=\"sam31-cgh\",
        name=None,
        dir=None,
        mode=None,
        config=None,
        **init_kwargs,
    ):
        _, self._rank = get_machine_local_and_dist_rank()
        self._run = None
        if self._rank != 0:
            return
        import wandb

        mode = mode or os.environ.get(\"WANDB_MODE\", \"online\")
        self._run = wandb.init(
            project=project,
            name=name,
            dir=dir,
            mode=mode,
            config=config,
            **init_kwargs,
        )
        atexit.register(self.close)

    def _to_wandb_value(self, value):
        if hasattr(value, \"detach\"):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
        return value

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self._run is None:
            return
        self._run.log(
            {key: self._to_wandb_value(value) for key, value in payload.items()},
            step=step,
        )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._run is None:
            return
        self._run.log({name: self._to_wandb_value(data)}, step=step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self._run is None:
            return
        if hparams:
            self._run.config.update(hparams, allow_val_change=True)
        if meters:
            self.log_dict(meters, step=0)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def make_wandb_logger(**kwargs):
    return WandBLogger(**kwargs)


class Logger:
    \"\"\"
    A logger class that can interface with TensorBoard and W&B loggers.
    \"\"\"
""",
        """        self.tb_logger = instantiate(tb_config) if tb_should_log else None
""": """        self.tb_logger = instantiate(tb_config) if tb_should_log else None
        wb_config = logging_conf.wandb_writer
        self.wb_logger = instantiate(wb_config) if wb_config else None
""",
        """        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
""": """        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if self.wb_logger:
            self.wb_logger.log_dict(payload, step)
""",
        """        if self.tb_logger:
            self.tb_logger.log(name, data, step)
""": """        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if self.wb_logger:
            self.wb_logger.log(name, data, step)
""",
        """        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
""": """        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        if self.wb_logger:
            self.wb_logger.log_hparams(hparams, meters)
""",
    }
    changed = False
    for old, new in replacements.items():
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"Could not find expected logger block in {path}")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print("patched", path)
    else:
        print("logger.py already patched")


def patch_config(sam3_repo: Path) -> None:
    path = (
        sam3_repo
        / "sam3"
        / "train"
        / "configs"
        / "cgh_pathology"
        / "cgh_pathology_sam31_seg.yaml"
    )
    if not path.exists():
        print("config not found yet, skipping", path)
        return
    text = path.read_text()
    text = text.replace(
        "sam3.losses.loss_fns.SemanticSegCriterion",
        "sam3.train.loss.loss_fns.SemanticSegCriterion",
    )
    text = text.replace("num_train_workers: 4", "num_train_workers: 0")
    text = text.replace("num_val_workers: 2", "num_val_workers: 0")
    text = text.replace("amp_dtype: bfloat16", "amp_dtype: float16")
    text = text.replace("resolution: 512", "resolution: 1008")
    text = text.replace("\n  max_ann_per_img: 200", "\n  max_ann_per_img: 50")
    text = text.replace("\n        max_ann_per_img: 500000", "\n        max_ann_per_img: 200")
    text = text.replace("\n        max_ann_per_img: 100000", "\n        max_ann_per_img: 200")
    text = text.replace("max_train_queries: 50000", "max_train_queries: 1024")
    text = text.replace("max_val_queries: 50000", "max_val_queries: 1024")
    text = text.replace(
        "    wandb_writer: null",
        """    wandb_writer:
      _target_: sam3.train.utils.logger.make_wandb_logger
      project: ${oc.env:WANDB_PROJECT,sam31-cgh}
      name: ${oc.env:WANDB_RUN_NAME,cgh_pathology_sam31}
      dir: ${launcher.experiment_log_dir}/wandb
      mode: ${oc.env:WANDB_MODE,online}""",
    )
    text = text.replace(
        """  val_transforms:
  - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
    transforms:
    - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
""",
        """  val_transforms:
  - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
    transforms:
    - _target_: sam3.train.transforms.segmentation.DecodeRle
    - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
""",
    )
    text = text.replace(
        """  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
""",
        """  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.segmentation.DecodeRle
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
""",
    )
    path.write_text(text)
    print("patched", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam3-repo",
        type=Path,
        default=PACKAGE_ROOT.parent / "sam3",
        help="Path to the facebookresearch/sam3 checkout.",
    )
    args = parser.parse_args()
    sam3_repo = args.sam3_repo.resolve()
    patch_vitdet(sam3_repo)
    patch_model_builder(sam3_repo)
    patch_optimizer(sam3_repo)
    patch_logger(sam3_repo)
    patch_config(sam3_repo)


if __name__ == "__main__":
    main()
