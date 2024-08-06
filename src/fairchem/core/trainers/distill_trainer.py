"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pathlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.utils as ptg_utils
from torch_scatter import scatter
from tqdm import tqdm

import wandb
from fairchem.core.common import distutils
from fairchem.core.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    OCPDistributedDataParallel,
    ParallelCollater,
)
from fairchem.core.common.registry import registry
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.common.relaxation.ml_relaxation import ml_relax
from fairchem.core.common.transforms import AddNoise, RandomJitter
from fairchem.core.common.utils import check_traj_files, irreps_sum, cg_change_mat
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.normalizer import Normalizer
from fairchem.core.trainers.base_trainer import BaseTrainer
from fairchem.core.models.equiformer_v2.trainers.forces_trainer import EquiformerV2ForcesTrainer


def vector_projection(vec1, vec2):
    # project vec1 on vec2
    vec2_normalized = torch.nn.functional.normalize(vec2, dim=-1)
    coef = torch.bmm(
        vec1.unsqueeze(-2), vec2_normalized.unsqueeze(-1)
    ).squeeze(-1)
    return coef * vec2_normalized


def get_h(n):
    return torch.eye(n) - 1 / n * torch.ones((n, n))


def center_matrix(M):
    n = M.shape[0]
    H = get_h(n).to(device=M.device)
    return torch.mm(H, torch.mm(M, H))


def gram_matrix(M):
    return torch.mm(M, M.t())


def hsic_given_centered(K_prime, L_prime):
    m = K_prime.shape[0]
    K_vec = K_prime.flatten().unsqueeze(-1)
    L_vec = L_prime.flatten().unsqueeze(-1)
    return (torch.mm(K_vec.t(), L_vec) / ((m - 1) ** 2)).squeeze()


def hsic(K, L):
    assert L.shape[0] == K.shape[0], "K and L have non-conforming sizes"
    K_prime = center_matrix(K)
    L_prime = center_matrix(L)
    return hsic_given_centered(K_prime, L_prime)


def cka(X, Y):
    """
    X: batch of features of shape BxD1
    Y: batch of features of shape BxD2
    """
    assert len(X.shape) == 2, "X matrix not correct shape"
    assert len(Y.shape) == 2, "Y matrix not correct shape"
    assert Y.shape[0] == X.shape[0], "X and Y doesn't have same batch size"
    with torch.cuda.amp.autocast(False):
        K = gram_matrix(X)
        L = gram_matrix(Y)

        # Center matrix here to avoid doing it multiple times
        K_prime = center_matrix(K)
        L_prime = center_matrix(L)
        kl = hsic_given_centered(K_prime, L_prime)
        kk = hsic_given_centered(K_prime, K_prime)
        ll = hsic_given_centered(L_prime, L_prime)
        return kl / torch.sqrt(kk * ll)


@registry.register_trainer("distill")
class DistillForcesTrainer(EquiformerV2ForcesTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_functions,
        evaluation_metrics,
        identifier,
        teacher_model,
        teacher_path,
        distillation,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm=None,
        noddp=False,
        name="distill",
        gp_gpus=None,
         **kwargs
    ):
        if slurm is None:
            slurm = {}
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_functions=loss_functions,
            evaluation_metrics=evaluation_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
            gp_gpus=gp_gpus,
        )
        # TODO: the way using config is quite strange. Clean the code.
        teacher_config = teacher_model
        teacher_model = teacher_config.pop("name")
        teacher_model_attributes = teacher_config
        self.config["teacher_model_attributes"] = teacher_model_attributes
        self.config["distillation"] = distillation
        
        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get("num_gaussians", 50)
        
        loader = self.train_loader or self.val_loader or self.test_loader
        self.teacher = registry.get_model_class(teacher_model)(
            loader.dataset[0].x.shape[-1]
            if loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            1,
            **teacher_model_attributes,
        ).to(self.device)
        self.teacher = OCPDataParallel(
            self.teacher,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.teacher = OCPDistributedDataParallel(
                self.teacher, device_ids=[self.device]
            )
        self.load_teacher(teacher_path)
        self.teacher.eval()
        if "random_jitter" in self.config["distillation"]["distill_loss"]:
            self.random_std = self.config["distillation"].get(
                "random_std", 0.1
            )
            if self.config["distillation"].get("random_fixed_length", False):
                self.random_fixed_length = self.config["distillation"].get(
                    "adversarial_alpha", 0.1
                )
            else:
                self.random_fixed_length = False
            self.random_mode = self.config["distillation"].get(
                "random_mode", None
            )
            self.transform = AddNoise()

        elif (
            "adversarial_jitter" in self.config["distillation"]["distill_loss"]
        ):
            self.transform = AddNoise()
            self.adversarial_lr = self.config["distillation"].get(
                "adversarial_lr", 0.1
            )
            self.n_adversarial_steps = self.config["distillation"].get(
                "n_adversarial_steps", 100
            )
            self.adversarial_alpha = self.config["distillation"].get(
                "adversarial_alpha", 0.1
            )
            self.adversarial_pgd = self.config["distillation"].get(
                "enable_pgd", False
            )
            self.adversarial_pgd_ball = (
                self.config["distillation"].get("pgd_ball", False)
                and self.adversarial_pgd
            )
            self.adversarial_pgd_mode = self.config["distillation"].get(
                "pgd_mode", None
            )
            self.adversarial_init_sd = self.config["distillation"].get(
                "adversarial_init_sd", 0.1
            )
            self.adversarial_teacher_grad = self.config["distillation"].get(
                "adversarial_teacher_grad", True
            )
            self.adversarial_force_prop = self.config["distillation"].get(
                "adversarial_force_prop", "prop"
            )
            self.force_regularization_lambda = self.config["distillation"].get(
                "force_reg", 0.0
            )
        self.v2v_geom_lambda = self.config["distillation"].get(
            "v2v_geom_lambda", 0.5
        )
        assert (
            self.v2v_geom_lambda <= 1.0 and self.v2v_geom_lambda >= 0.0
        ), "distillation.v2v_geom_lambda must be between 0 and 1"
        if self.config["logger"]["name"] == "wandb" and distutils.is_master():
            wandb.config.update({"distillation": self.config["distillation"]})

        self.distill_fns = [
            dist_fn.strip()
            for dist_fn in self.config["distillation"]["distill_loss"].split(
                ","
            )
        ]
        if isinstance(self.config["distillation"]["distill_lambda"], float):
            self.distill_lambda = [
                self.config["distillation"]["distill_lambda"]
            ] * len(self.distill_fns)
        else:
            self.distill_lambda = self.config["distillation"]["distill_lambda"]

        if (
            "adversarial" in self.config["distillation"]["distill_loss"]
            or "random_jitter" in self.config["distillation"]["distill_loss"]
        ):
            self.adversarial_distill_fns = [
                ad_fn.strip()
                for ad_fn in self.config["distillation"][
                    "adversarial_distill_loss"
                ].split(",")
            ]
            if isinstance(
                self.config["distillation"]["adversarial_distill_lambda"],
                float,
            ):
                self.adversarial_distill_lambda = [
                    self.config["distillation"]["adversarial_distill_lambda"]
                ] * len(self.distill_fns)
            else:
                self.adversarial_distill_lambda = self.config["distillation"][
                    "adversarial_distill_lambda"
                ]
        self.use_mae = self.config["distillation"].get("use_mae", False)
        self.use_huber = self.config["distillation"].get("use_huber", False)
        self.huber_delta = self.config["distillation"].get("huber_delta", 1.0)

    def load_teacher(self, checkpoint_path):
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        # Load model, optimizer, normalizer state dict.
        # if trained with ddp and want to load in non-ddp, modify keys from
        # module.module.. -> module..
        first_key = next(iter(checkpoint["state_dict"]))
        strict = self.config["task"].get("strict", True)
        if (
            not distutils.initialized() or self.config["noddp"]
        ) and first_key.split(".")[1] == "module":
            # No need for OrderedDict since dictionaries are technically ordered
            # since Python 3.6 and officially ordered since Python 3.7
            new_dict = {k[7:] : v for k, v in checkpoint["state_dict"].items()}
            self.teacher.load_state_dict(new_dict, strict= strict)
        elif distutils.initialized() and first_key.split(".")[1] != "module":
            new_dict = {
                f"module.{k}": v for k, v in checkpoint["state_dict"].items()
            }
            self.teacher.load_state_dict(new_dict,)
        else:
            self.teacher.load_state_dict(checkpoint["state_dict"])

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: str | None = None,
        disable_tqdm: bool = False,
    ):
        if self.is_debug and per_image:
            raise FileNotFoundError("Predictions require debug mode to be turned off.")

        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        predictions = defaultdict(list)

        for _i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            for target_key in self.config["outputs"]:
                pred = out[target_key]
                if self.normalizers.get(target_key, False):
                    pred = self.normalizers[target_key].denorm(pred)

                if per_image:
                    ### Save outputs in desired precision, default float16
                    if (
                        self.config["outputs"][target_key].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get("prediction_dtype", "float16")
                        == "float32"
                        or self.config["task"].get("dataset", "lmdb") == "oc22_lmdb"
                    ):
                        dtype = torch.float32
                    else:
                        dtype = torch.float16

                    pred = pred.cpu().detach().to(dtype)
                    ### Split predictions into per-image predictions
                    if self.config["outputs"][target_key]["level"] == "atom":
                        batch_natoms = batch.natoms
                        batch_fixed = batch.fixed
                        per_image_pred = torch.split(pred, batch_natoms.tolist())

                        ### Save out only free atom, EvalAI does not need fixed atoms
                        _per_image_fixed = torch.split(
                            batch_fixed, batch_natoms.tolist()
                        )
                        _per_image_free_preds = [
                            _pred[(fixed == 0).tolist()].numpy()
                            for _pred, fixed in zip(per_image_pred, _per_image_fixed)
                        ]
                        _chunk_idx = np.array(
                            [free_pred.shape[0] for free_pred in _per_image_free_preds]
                        )
                        per_image_pred = _per_image_free_preds
                    ### Assumes system level properties are of the same dimension
                    else:
                        per_image_pred = pred.numpy()
                        _chunk_idx = None

                    predictions[f"{target_key}"].extend(per_image_pred)
                    ### Backwards compatibility, retain 'chunk_idx' for forces.
                    if _chunk_idx is not None:
                        if target_key == "forces":
                            predictions["chunk_idx"].extend(_chunk_idx)
                        else:
                            predictions[f"{target_key}_chunk_idx"].extend(_chunk_idx)
                else:
                    predictions[f"{target_key}"] = pred.detach()

            if not per_image:
                return predictions

            ### Get unique system identifiers
            sids = (
                batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid
            )
            ## Support naming structure for OC20 S2EF
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

        self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions
    
    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    def _global_preservation(self, feat_s, feat_t, batch_list):
        num_atoms_per_image = torch.cat([b.natoms for b in batch_list], dim=0)
        num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

        # The following is borrowed from common.utils.radius_graph_pbc
        # index offset between images
        index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
        )

        index_offset_expand = torch.repeat_interleave(
            index_offset, num_atoms_per_image_sqr
        )
        num_atoms_per_image_expand = torch.repeat_interleave(
            num_atoms_per_image, num_atoms_per_image_sqr
        )

        # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
        # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
        # the following (but 10x faster since it removes the for loop)
        # for batch_idx in range(batch_size):
        #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
        num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
        index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0)
            - num_atoms_per_image_sqr
        )
        index_sqr_offset = torch.repeat_interleave(
            index_sqr_offset, num_atoms_per_image_sqr
        )
        atom_count_sqr = (
            torch.arange(num_atom_pairs, device=self.device) - index_sqr_offset
        )

        # Compute the indices for the pairs of atoms (using division and mod)
        # If the systems get too large this apporach could run into numerical precision issues
        index1 = (
            atom_count_sqr // num_atoms_per_image_expand
        ) + index_offset_expand
        index2 = (
            atom_count_sqr % num_atoms_per_image_expand
        ) + index_offset_expand

        feat_s1 = torch.index_select(feat_s, 0, index1)
        feat_s2 = torch.index_select(feat_s, 0, index2)
        feat_t1 = torch.index_select(feat_t, 0, index1)
        feat_t2 = torch.index_select(feat_t, 0, index2)

        feat_s_distances = torch.mean(
            F.mse_loss(feat_s1, feat_s2, reduction="none"), dim=1
        )
        feat_t_distances = torch.mean(
            F.mse_loss(feat_t1, feat_t2, reduction="none"), dim=1
        )
        while len(feat_s_distances.shape) > 1:
            feat_s_distances = torch.mean(feat_s_distances, dim=1)
            feat_t_distances = torch.mean(feat_t_distances, dim=1)
        dist = F.mse_loss(feat_s_distances, feat_t_distances, reduction="none")
        loss = scatter(dist, index1, dim=0, reduce="mean")
        loss = scatter(loss, batch_list.batch, dim=0, reduce="mean")
        return torch.mean(loss)

    def _local_preservation(self, feat_s, feat_t, edge_index):
        index1, index2 = edge_index
        s_sim = ptg_utils.softmax(
            (feat_s[index1] - feat_s[index2]).norm(p=2, dim=-1), index2
        )
        t_sim = ptg_utils.softmax(
            (feat_t[index1] - feat_t[index2]).norm(p=2, dim=-1), index2
        )

        lsp_loss = F.kl_div(torch.log(s_sim), t_sim, log_target=False)
        return lsp_loss

    def _node_local_preservation_distill_loss(self, out_batch, batch):
        return self._local_preservation(
            out_batch["out"]["node_feature"],
            out_batch["t_out"]["node_feature"],
            batch[0].edge_index,
        )

    def _edge_local_preservation_distill_loss(self, out_batch, batch):
        return self._local_preservation(
            out_batch["out"]["n2e_feature"],
            out_batch["t_out"]["e2n_feature"],
            batch[0].edge_index,
        )

    def _node_cka_distill_loss(self, out_batch, batch):
        cka_loss = 1 - cka(
            out_batch["out"]["node_feature"],
            out_batch["t_out"]["node_feature"],
        )
        return cka_loss

    def _node_global_preservation_distill_loss(self, out_batch, batch):
        return self._global_preservation(
            out_batch["out"]["node_feature"],
            out_batch["t_out"]["node_feature"],
            batch,
        )

    def _edge_global_preservation_distill_loss(self, out_batch, batch):
        return self._global_preservation(
            out_batch["out"]["n2e_feature"],
            out_batch["t_out"]["e2n_feature"],
            batch,
        )

    def _vec_global_preservation_distill_loss(self, out_batch, batch):
        return self._global_preservation(
            out_batch["out"]["vector_feature"],
            out_batch["t_out"]["vector_feature"],
            batch,
        )

    def _node2node_distill_loss(self, out_batch, batch):
        if out_batch["out"]["node_feature"].shape == out_batch["t_out"]["node_feature"].shape:
            teacher_node_features = out_batch["t_out"]["node_feature"]
        else:
            # for eqv2-eqv2(small) distillatio select the first 4 spherical channels up to L=1 and M=1 
            teacher_node_features = out_batch["t_out"]["node_feature"].narrow(1,0,4)
        
        if self.use_mae:
           return torch.nn.functional.l1_loss(
                out_batch["out"]["node_feature"],
                teacher_node_features,
            )
        elif self.use_huber:
            return torch.nn.functional.huber_loss(
                out_batch["out"]["node_feature"],
                teacher_node_features,
                delta=self.huber_delta,
            )
        else:
            return torch.nn.functional.mse_loss(
                out_batch["out"]["node_feature"],
                teacher_node_features,
            )

    def _edge2node_distill_loss(self, out_batch, batch):
        if self.use_mae:
            return torch.nn.functional.l1_loss(
                out_batch["out"]["n2e_feature"],
                out_batch["t_out"]["e2n_feature"],
            )
        elif self.use_huber:
            return torch.nn.functional.huber_loss(
                out_batch["out"]["n2e_feature"],
                out_batch["t_out"]["e2n_feature"],
                delta=self.huber_delta,
            )
        else:
            return torch.nn.functional.mse_loss(
                out_batch["out"]["n2e_feature"],
                out_batch["t_out"]["e2n_feature"],
            )

    def _edge2edge_distill_loss(self, out_batch, batch):
        if out_batch["out"]["e2e_feature"].shape == out_batch["t_out"]["edge_feature"].shape:
            teacher_node_features = out_batch["t_out"]["edge_feature"]
        else:
            # for eqv2-eqv2(small) distillatio select the first 4 spherical channels up to L=1 and M=1 
            teacher_node_features = out_batch["t_out"]["edge_feature"].narrow(1,0,4)
        
        return torch.nn.functional.mse_loss(
            out_batch["out"]["e2e_feature"], teacher_node_features
        )

    def _per_atom_energy_distill_loss(self, out_batch, batch):
        return self.loss_fn["energy"](
            out_batch["out"]["per_atom_energy"],
            out_batch["t_out"]["per_atom_energy"],
        )

    def _vec2vec_distill_loss(self, out_batch, batch):
        return torch.nn.functional.mse_loss(
            out_batch["out"]["vector_feature"],
            out_batch["t_out"]["vector_feature"],
        )

    def _vec2vec_geometric(self, out_batch, batch):
        dir_loss = 1 - F.cosine_similarity(
            out_batch["out"]["vector_feature"],
            out_batch["t_out"]["vector_feature"],
            1,
        )
        dir_loss = torch.mean(dir_loss, dim=1)
        dir_loss = torch.mean(
            scatter(dir_loss, batch[0].batch, dim=0, reduce="mean")
        )

        norm_loss = F.l1_loss(
            torch.linalg.norm(out_batch["out"]["vector_feature"], dim=1),
            torch.linalg.norm(out_batch["t_out"]["vector_feature"], dim=1),
            reduction="none",
        )
        norm_loss = torch.mean(norm_loss, dim=1)
        norm_loss = torch.mean(
            scatter(norm_loss, batch[0].batch, dim=0, reduce="mean")
        )
        return (
            1 - self.v2v_geom_lambda
        ) * norm_loss + self.v2v_geom_lambda * dir_loss

    def _loss_weights_d1M(self, batch, loss_type="main"):
        """
        loss_type: str
            "main" or "distill" to distinguist between the main loss and the
            distillation loss. At this stage we hard-code "main" to consider
            only ground truth samples (no synthetic samples) and "distill" to
            weight both terms with the loss_weighting_synthetic parameter.
        """

        if not (
            ("distillation" in self.config)
            and ("loss_weighting_synthetic" in self.config["distillation"])
        ):
            # This step is redundand with the code below, but for the sake of
            # better readability we set the values to 1 (disabling any
            # weighting) before we do any computation if the relevant
            # parameter `loss_weighting_synthetic` cannot be found.
            w_per_node = torch.ones_like(batch[0].tags)
            w_per_sample = torch.ones_like(batch[0].energy)

            return w_per_node, w_per_sample

        ratio_synth_to_dft = self.config["distillation"].get(
            "loss_weighting_synthetic"
        )
        assert ratio_synth_to_dft > 0.0

        # The systems in the original OC20 dataset was drawn with random
        # numbers between 0...2,499,999, and for our synthetic data we used
        # seeds between 5,000,000...5,099,999. To distinguist the origin of a
        # sample (frame along a trajectory) we simply use the seed information
        # Note: to the best of my knowledge the `seed` in the system sampling
        # procedure corresponds to the `sid` (system ID) of a datapoint.

        mask_synth_systems_bool = batch[0].sid > 4999999
        mask_synth_systems_int = torch.where(mask_synth_systems_bool, 1.0, 0.0)
        batch_ratio_synth = mask_synth_systems_int.mean()
        if loss_type == "distill":
            w_dft = 1 / (
                1 - batch_ratio_synth + ratio_synth_to_dft * batch_ratio_synth
            )
            w_s = ratio_synth_to_dft * w_dft
        elif loss_type == "main":
            w_dft = 1 / (1 - batch_ratio_synth)
            w_s = 0.0 * w_dft

        synth_idx = mask_synth_systems_bool.nonzero().squeeze()
        mask_synth_per_node = torch.isin(batch[0].batch, synth_idx)
        weights_per_node = torch.where(mask_synth_per_node, w_s, w_dft)
        weights_per_sample = torch.where(mask_synth_systems_bool, w_s, w_dft)

        return weights_per_node, weights_per_sample

    def _vec2vec_distill_loss_d1M(self, out_batch, batch):

        w_per_node, _ = self._loss_weights_d1M(batch, loss_type="distill")
        # using the MSE loss we undo the square here
        w = torch.sqrt(w_per_node)[:, None, None]

        return torch.nn.functional.mse_loss(
            out_batch["out"]["vector_feature"] * w,
            out_batch["t_out"]["vector_feature"] * w,
        )

    def _node2node_distill_loss_d1M(self, out_batch, batch):

        w_per_node, _ = self._loss_weights_d1M(batch, loss_type="distill")
        # using the MSE loss we undo the square here
        w = torch.sqrt(w_per_node)[:, None]

        return torch.nn.functional.mse_loss(
            out_batch["out"]["node_feature"] * w,
            out_batch["t_out"]["node_feature"] * w,
        )

    def _adversarial_batch(self, batch_list):
        with torch.no_grad():
            if self.adversarial_init_sd > 0:
                delta_list = [
                    torch.empty(
                        batch.pos.shape, requires_grad=True, device=self.device
                    ).normal_(0, self.adversarial_init_sd)
                    for batch in batch_list
                ]
            else:
                delta_list = [
                    torch.zeros(
                        batch.pos.shape, requires_grad=True, device=self.device
                    )
                    for batch in batch_list
                ]
        opt = optim.Adam(delta_list, lr=self.adversarial_lr)
        for i in range(self.n_adversarial_steps):
            opt.zero_grad()
            batch_list_noise = [
                self.transform(batch.clone(), delta)
                for batch, delta in zip(batch_list, delta_list)
            ]
            out_batch = self._distill_forward(batch_list_noise)
            loss = -self._compute_loss(
                out_batch["out"], batch_list_noise, out_batch["t_out"]
            )  # minimize negative loss <=> maximize loss
            loss.backward()
            opt.step()
            # TODO: do we really need this?
            with torch.no_grad():
                return_batch = [
                    self.transform(batch.clone(), delta)
                    for batch, delta in zip(batch_list, delta_list)
                ]
        return [batch.detach() for batch in return_batch]

    def _adversarial_pgd_batch(self, batch_list):
        with torch.no_grad():
            if self.adversarial_init_sd > 0:
                delta_list = [
                    torch.empty(
                        batch.pos.shape, requires_grad=True, device=self.device
                    ).normal_(0, self.adversarial_init_sd)
                    for batch in batch_list
                ]
            else:
                delta_list = [
                    torch.zeros(
                        batch.pos.shape, requires_grad=True, device=self.device
                    )
                    for batch in batch_list
                ]
        for i in range(self.n_adversarial_steps):
            batch_list_noise = [
                self.transform(batch.clone(), delta)
                for batch, delta in zip(batch_list, delta_list)
            ]

            out_batch = self._distill_forward(
                batch_list_noise, teacher_grad=self.adversarial_teacher_grad
            )
            loss = 0.0
            for loss_idx, loss_type in enumerate(self.adversarial_distill_fns):
                if loss_type == "regular":
                    loss += (
                        self._compute_loss_distill(
                            out_batch["out"],
                            batch_list_noise,
                            out_batch["t_out"],
                        )
                        * self.adversarial_distill_lambda[loss_idx]
                    )
                else:
                    loss += (
                        getattr(self, "_" + loss_type)(
                            out_batch, batch_list_noise
                        )
                        * self.adversarial_distill_lambda[loss_idx]
                    )
            if self.force_regularization_lambda > 0.0:
                if not self.adversarial_teacher_grad:
                    _, t_out_forces = self.teacher(batch_list_noise)
                    loss -= self.force_regularization_lambda * torch.mean(
                        torch.linalg.norm(t_out_forces, dim=1)
                    )
                else:
                    loss -= self.force_regularization_lambda * torch.mean(
                        torch.linalg.norm(out_batch["t_out"]["forces"], dim=1)
                    )
            torch.autograd.backward([loss], inputs=delta_list)
            for j in range(len(delta_list)):
                with torch.no_grad():
                    if self.adversarial_pgd_mode == "ball":
                        gradient = self.adversarial_lr * delta_list[j].grad
                        mask = (
                            torch.linalg.norm(gradient, dim=1)
                            > self.adversarial_alpha
                        )
                        gradient[mask] = self.adversarial_alpha * F.normalize(
                            gradient[mask]
                        )
                        delta_list[j] += gradient
                    elif self.adversarial_pgd_mode == "force_proj":
                        proj = vector_projection(
                            delta_list[j].grad, batch_list_noise[j].force
                        )
                        displacement = delta_list[j].grad - proj
                        if self.adversarial_force_prop == "prop":
                            force_norm = torch.linalg.norm(
                                batch_list_noise[j].force, dim=1
                            ).unsqueeze(-1)
                            norm = self.adversarial_lr * force_norm
                        elif self.adversarial_force_prop == "inv_prop":
                            force_norm = torch.linalg.norm(
                                batch_list_noise[j].force, dim=1
                            ).unsqueeze(-1)
                            norm = self.adversarial_lr / force_norm
                        else:
                            norm = self.adversarial_alpha
                        delta_list[j] += norm * F.normalize(displacement)

                    else:  # PGD sphere
                        delta_list[j] += self.adversarial_alpha * F.normalize(
                            delta_list[j].grad
                        )

        batch_list_noise = [
            self.transform(batch.clone(), delta)
            for batch, delta in zip(batch_list, delta_list)
        ]
        return [batch.detach() for batch in batch_list_noise]

    def _adversarial_jitter_distill_loss(self, out_batch, batch):
        self.model.eval()
        if self.adversarial_pgd:
            augmented_batch = self._adversarial_pgd_batch(batch)
        else:
            augmented_batch = self._adversarial_batch(batch)
        self.model.train()
        out_batch = self._distill_forward(augmented_batch)

        distill_loss = 0.0
        for loss_idx, loss_type in enumerate(self.adversarial_distill_fns):
            if loss_type == "regular":
                distill_loss += (
                    self._compute_loss_distill(
                        out_batch["out"], augmented_batch, out_batch["t_out"]
                    )
                    * self.adversarial_distill_lambda[loss_idx]
                )
            else:
                distill_loss += (
                    getattr(self, "_" + loss_type)(out_batch, augmented_batch)
                    * self.adversarial_distill_lambda[loss_idx]
                )
        return distill_loss

    def _random_jitter_batch(self, batch_list):
        with torch.no_grad():
            delta_list = [
                torch.zeros(
                    batch.pos.shape, requires_grad=False, device=self.device
                )
                for batch in batch_list
            ]
            for j in range(len(delta_list)):
                displacement = torch.empty(
                    batch_list[j].pos.shape,
                    requires_grad=False,
                    device=self.device,
                ).normal_(0, self.random_std)
                if self.random_mode == "force_proj":
                    proj = vector_projection(displacement, batch_list[j].force)
                    displacement = displacement - proj
                if self.random_mode == "proj_on_force":
                    displacement = vector_projection(
                        displacement, batch_list[j].force
                    )
                if self.random_mode == "sample_from_force":
                    displacement = torch.normal(
                        batch_list[j].force, self.random_std
                    )
                # potentially fix length
                if self.random_fixed_length:
                    delta_list[j] += self.random_fixed_length * F.normalize(
                        displacement
                    )
                else:
                    delta_list[j] += displacement
        batch_list_noise = [
            self.transform(batch.clone(), delta)
            for batch, delta in zip(batch_list, delta_list)
        ]
        return [batch.detach() for batch in batch_list_noise]

    def _random_jitter_distill_loss(self, out_batch, batch):
        augmented_batch = self._random_jitter_batch(batch)
        # out_batch = self._distill_forward_energy_forces_only(augmented_batch)
        out_batch = self._distill_forward(augmented_batch)
        distill_loss = 0.0
        for loss_idx, loss_type in enumerate(self.adversarial_distill_fns):
            if loss_type == "regular":
                distill_loss += (
                    self._compute_loss_distill(
                        out_batch["out"], augmented_batch, out_batch["t_out"]
                    )
                    * self.adversarial_distill_lambda[loss_idx]
                )
            else:
                distill_loss += (
                    getattr(self, "_" + loss_type)(out_batch, augmented_batch)
                    * self.adversarial_distill_lambda[loss_idx]
                )
        return distill_loss

    def train(self, disable_eval_tqdm=False):  # noqa: C901
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out_batch = self._distill_forward(batch)
                    loss = self._compute_loss(out_batch["out"], batch)
                    distill_loss = []
                    for loss_idx, loss_type in enumerate(self.distill_fns):
                        if loss_type == "regular":
                            distill_loss.append(
                                self._compute_loss_distill(
                                    out_batch["out"],
                                    batch,
                                    out_batch["t_out"],
                                )
                                * self.distill_lambda[loss_idx]
                            )
                        else:
                            distill_loss.append(
                                getattr(self, "_" + loss_type)(
                                    out_batch, batch
                                )
                                * self.distill_lambda[loss_idx]
                            )
                    loss += sum(distill_loss)

                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out_batch["out"],
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )
                for idx, loss_i in enumerate(distill_loss):
                    self.metrics = self.evaluator.update(
                        f"distill_loss_{idx}", loss_i.item(), self.metrics
                    )
                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    # and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )
                        # if self.is_hpo:
                        #     self.hpo_update(
                        #         self.epoch,
                        #         self.step,
                        #         self.metrics,
                        #         val_metrics,
                        #     )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        out = self.model(batch_list.to(self.device))

        ### TODO: Move into BaseModel in OCP 2.0
        outputs = {}
        batch_size = batch_list.natoms.numel()
        num_atoms_in_batch = batch_list.natoms.sum()
        for target_key in self.output_targets:
            ### Target property is a direct output of the model
            if target_key in out:
                pred = out[target_key]
            ## Target property is a derived output of the model. Construct the
            ## parent property
            else:
                _max_rank = 0
                for subtarget_key in self.output_targets[target_key]["decomposition"]:
                    _max_rank = max(
                        _max_rank,
                        self.output_targets[subtarget_key]["irrep_dim"],
                    )

                pred_irreps = torch.zeros(
                    (batch_size, irreps_sum(_max_rank)), device=self.device
                )

                for subtarget_key in self.output_targets[target_key]["decomposition"]:
                    irreps = self.output_targets[subtarget_key]["irrep_dim"]
                    _pred = out[subtarget_key]

                    if self.normalizers.get(subtarget_key, False):
                        _pred = self.normalizers[subtarget_key].denorm(_pred)

                    ## Fill in the corresponding irreps prediction
                    ## Reshape irrep prediction to (batch_size, irrep_dim)
                    pred_irreps[
                        :,
                        max(0, irreps_sum(irreps - 1)) : irreps_sum(irreps),
                    ] = _pred.view(batch_size, -1)

                pred = torch.einsum(
                    "ba, cb->ca",
                    cg_change_mat(_max_rank, self.device),
                    pred_irreps,
                )

            ### not all models are consistent with the output shape
            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_key]["level"] == "atom":
                pred = pred.view(num_atoms_in_batch, -1)
            else:
                pred = pred.view(batch_size, -1)

            outputs[target_key] = pred

        return outputs

    def _distill_forward_energy_forces_only(self, batch_list):
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_forces = self.model(batch_list)
            with torch.no_grad():
                t_out_energy, t_out_forces = self.teacher(batch_list)
        else:
            out_energy = self.model(batch_list)
            with torch.no_grad():
                t_out_energy = self.teacher(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)
        if t_out_energy.shape[-1] == 1:
            t_out_energy = t_out_energy.view(-1)

        out = {"energy": out_energy}
        t_out = {"energy": t_out_energy}

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        if self.config["teacher_model_attributes"].get("regress_forces", True):
            t_out["forces"] = t_out_forces
        return {"out": out, "t_out": t_out}

    def _distill_forward(self, batch_list, teacher_grad=False):
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            if not teacher_grad:
                with torch.no_grad():
                    (
                        [tfnode, tfe2n, tfvec, tfedge],
                        [t_out_energy, t_out_forces, t_out_per_atom_energy],
                        main_graph,
                    ) = self.teacher.extract_features(batch_list)
            else:
                (
                    [tfnode, tfe2n, tfvec, tfedge],
                    [t_out_energy, t_out_forces, t_out_per_atom_energy],
                    main_graph,
                ) = self.teacher.extract_features(batch_list)
            if "edge2edge_distill_loss" not in self.distill_fns:
                main_graph = None
            (
                [sfnode, sfn2e, sfvec, sfe2e],
                [
                    out_energy,
                    out_forces,
                    out_per_atom_energy,
                ],
                _,
            ) = self.model.extract_features((batch_list, main_graph))

        else:
            [sfnode, sfn2e, sfvec], out_energy = self.model.extract_features(
                batch_list
            )
            if not teacher_grad:
                with torch.no_grad():
                    [
                        tfnode,
                        tfe2n,
                    ], t_out_energy = self.teacher.extract_features(batch_list)
            else:
                [
                    tfnode,
                    tfe2n,
                ], t_out_energy = self.teacher.extract_features(batch_list)

        # Don't know why these two lines are here, they just seem to makes thing go slower and require more memory?
        # with torch.no_grad():
        #    t_out = self.teacher.extract_features(batch_list)

        out = {
            "node_feature": sfnode,
            "n2e_feature": sfn2e,
            "vector_feature": sfvec,
            "e2e_feature": sfe2e,
            "energy": out_energy,
            "per_atom_energy": out_per_atom_energy,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        t_out = {
            "node_feature": tfnode,
            "e2n_feature": tfe2n,
            "vector_feature": tfvec,
            "edge_feature": tfedge,
            "energy": t_out_energy,
            "per_atom_energy": t_out_per_atom_energy,
        }

        if self.config["teacher_model_attributes"].get("regress_forces", True):
            t_out["forces"] = t_out_forces
        return {"out": out, "t_out": t_out}

    def _compute_loss(self, out, batch, teacher_output=None):
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0

        loss = []
        for loss_fn in self.loss_functions:
            target_name, loss_info = loss_fn

            target = batch[target_name]
            pred = out[target_name]

            natoms = batch.natoms
            natoms = torch.repeat_interleave(natoms, natoms)

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["train_on_free_atoms"]
            ):
                target = target[mask]
                pred = pred[mask]
                natoms = natoms[mask]

            num_atoms_in_batch = natoms.numel()
            if self.normalizers.get(target_name, False):
                target = self.normalizers[target_name].norm(target)
            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
                pred = pred.view(num_atoms_in_batch, -1)
                
            else:
                target = target.view(batch_size, -1)
                pred = pred.view(batch_size, -1)
            
            
            mult = loss_info["coefficient"]
            loss.append(
                mult
                * loss_info["fn"](
                    pred,
                    target,
                    natoms=natoms,
                    batch_size=batch_size,
                )
            )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        return sum(loss)

    def _compute_loss_distill(self, out, batch_list, teacher_output=None):
        loss = []

        # Energy loss.
        if teacher_output is not None:
            energy_target = teacher_output["energy"]
        else:
            energy_target = torch.cat(
                [batch.energy.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["distillation"].get(
            "energy_coefficient", 0.0
        )

        loss.append(
            energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
        )

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            if teacher_output is not None:
                force_target = teacher_output["forces"]
            else:
                force_target = torch.cat(
                    [batch.force.to(self.device) for batch in batch_list],
                    dim=0,
                )
                if self.normalizer.get("normalize_labels", False):
                    force_target = self.normalizers["grad_target"].norm(
                        force_target
                    )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [
                        batch.tags.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                loss_force_list = torch.abs(out["forces"] - force_target)
                train_loss_force_unnormalized = torch.sum(
                    loss_force_list * weight.view(-1, 1)
                )
                train_loss_force_normalizer = 3.0 * weight.sum()

                # add up normalizer to obtain global normalizer
                distutils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                    distutils.get_world_size() / train_loss_force_normalizer
                )
                loss.append(train_loss_force_normalized)

            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["distillation"].get(
                    "force_coefficient", 30
                )
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    if (
                        self.config["optim"]
                        .get("loss_force", "mae")
                        .startswith("atomwise")
                    ):
                        force_mult = self.config["optim"].get(
                            "force_coefficient", 1
                        )
                        natoms = torch.cat(
                            [
                                batch.natoms.to(self.device)
                                for batch in batch_list
                            ]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        force_loss = force_mult * self.loss_fn["force"](
                            out["forces"][mask],
                            force_target[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list.natoms.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        loss.append(
                            force_mult
                            * self.loss_fn["force"](
                                out["forces"][mask], force_target[mask]
                            )
                        )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"], force_target)
                    )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        if metrics is None:
            metrics = {}
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        fixed = batch.fixed
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        targets = {}
        for target_name in self.output_targets:
            target = batch[target_name]
            num_atoms_in_batch = batch.natoms.sum()

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["eval_on_free_atoms"]
            ):
                target = target[mask]
                out[target_name] = out[target_name][mask]
                num_atoms_in_batch = natoms.sum()
            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
                out[target_name] = out[target_name].view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)
                out[target_name] = out[target_name].view(batch_size, -1)

            targets[target_name] = target
            if self.normalizers.get(target_name, False):
                out[target_name] = self.normalizers[target_name].denorm(
                    out[target_name]
                )

        targets["natoms"] = natoms
        out["natoms"] = natoms
        return evaluator.eval(out, targets, prev_metrics=metrics)

    def run_relaxations(self, split="val"):
        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()
