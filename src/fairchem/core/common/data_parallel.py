"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numba
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import BatchSampler, DistributedSampler, Sampler
from torch.nn.parallel.distributed import DistributedDataParallel

from fairchem.core.common import distutils, gp_utils
from fairchem.core.datasets import data_list_collater

if TYPE_CHECKING:
    from pathlib import Path

    from torch_geometric.data import Batch, Data


class OCPCollater:
    def __init__(self, otf_graph: bool = False) -> None:
        self.otf_graph = otf_graph

    def __call__(self, data_list: list[Data]) -> Batch:
        return data_list_collater(data_list, otf_graph=self.otf_graph)


@numba.njit
def balanced_partition(sizes: npt.NDArray[np.int_], num_parts: int):
    """
    Greedily partition the given set by always inserting
    the largest element into the smallest partition.
    """
    sort_idx = np.argsort(-sizes)  # Sort in descending order
    heap: list[tuple[list[int], list[int]]] = [
        (sizes[idx], [idx]) for idx in sort_idx[:num_parts]
    ]
    heapq.heapify(heap)
    for idx in sort_idx[num_parts:]:
        smallest_part = heapq.heappop(heap)
        new_size = smallest_part[0] + sizes[idx]
        new_idx = smallest_part[1] + [idx]
        heapq.heappush(heap, (new_size, new_idx))
    return [part[1] for part in heap]


@runtime_checkable
class _HasMetadata(Protocol):
    @property
    def metadata_path(self) -> Path: ...


class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size, **kwargs):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset=dataset, **kwargs)

        self.start_iter = 0
        self.batch_size = batch_size
        assert self.batch_size > 0, "batch_size not set for the sampler"
        logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # TODO: For very large datasets, even virtual datasets this might slow down
        # or not work correctly. The issue is that we enumerate the full list of all
        # samples in a single epoch, and manipulate this list directly. A better way
        # of doing this would be to keep this sequence strictly as an iterator
        # that stores the current state (instead of the full sequence)
        distributed_sampler_sequence = super().__iter__()
        if self.start_iter > 0:
            for i, _ in enumerate(distributed_sampler_sequence):
                if i == self.start_iter * self.batch_size - 1:
                    break
        return distributed_sampler_sequence

    def set_epoch_and_start_iteration(self, epoch, start_iter):
        self.set_epoch(epoch)
        self.start_iter = start_iter


class BalancedBatchSampler(Sampler):
    def _load_dataset(self, dataset, mode: Literal["atoms", "neighbors"]):
        errors: list[str] = []
        if not isinstance(dataset, _HasMetadata):
            errors.append(f"Dataset {dataset} does not have a metadata_path attribute.")
            return None, errors
        if not dataset.metadata_path.exists():
            errors.append(f"Metadata file {dataset.metadata_path} does not exist.")
            return None, errors

        key = {"atoms": "natoms", "neighbors": "neighbors"}[mode]
        sizes = np.load(dataset.metadata_path)[key]

        return sizes, errors

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        device: torch.device,
        seed: int,
        mode: str | bool = "atoms",
        shuffle: bool = True,
        drop_last: bool = False,
        force_balancing: bool = False,
        throw_on_error: bool = False,
        distill_args=None,
    ) -> None:
        if mode is True:
            mode = "atoms"

        if isinstance(mode, str):
            mode = mode.lower()
            if mode not in ("atoms", "neighbors"):
                raise ValueError(
                    f"Invalid mode {mode}. Must be one of 'atoms', 'neighbors', or a boolean."
                )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.device = device
        self.mode = mode
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        
        if distill_args and ("target_ratio_synthetic" in distill_args):
            assert hasattr(dataset, "metadata_path"), "No metadata path found"
            assert dataset.metadata_path.is_file(), "Metadata is not a file"
            assert "origin" in np.load(
                dataset.metadata_path
            ), "Metadata lacks 'origin'. Run srcipts/merge_datasets.py"

            mask_origin = np.load(dataset.metadata_path)["origin"]
            trs = distill_args["target_ratio_synthetic"]
            # dataset_ratio_synthetic. Label=1 marks synthetic data
            drs = (mask_origin == 1).mean()

            # caution: these probabilities do not need to sum up to one
            prob_synthetic = trs / drs
            prob_dft = (1 - trs) / (1 - drs)
            weights = np.where(mask_origin == 1, prob_synthetic, prob_dft)

            self.single_sampler = DistributedWeightedSampler(
                dataset,
                weights=weights,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
            )
        else:
            self.single_sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
            )

        self.batch_sampler = BatchSampler(
            self.single_sampler,
            batch_size,
            drop_last=drop_last,
        )

        self.sizes = None
        self.balance_batches = False

        if self.num_replicas <= 1:
            logging.info("Batch balancing is disabled for single GPU training.")
            return

        if self.mode is False:
            logging.info(
                "Batch balancing is disabled because `optim.load_balancing` is `False`"
            )
            return

        self.sizes, errors = self._load_dataset(dataset, self.mode)
        if self.sizes is None:
            self.balance_batches = force_balancing
            if force_balancing:
                errors.append(
                    "BalancedBatchSampler has to load the data to  determine batch sizes, which incurs significant overhead! "
                    "You can disable balancing by setting `optim.load_balancing` to `False`."
                )
            else:
                errors.append(
                    "Batches will not be balanced, which can incur significant overhead!"
                )
        else:
            self.balance_batches = True

        if errors:
            msg = "BalancedBatchSampler: " + " ".join(errors)
            if throw_on_error:
                raise RuntimeError(msg)

            logging.warning(msg)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def set_epoch(self, epoch):
        self.single_sampler.set_epoch(epoch)

    def set_epoch_and_start_iteration(self, epoch: int, start_iteration: int) -> None:
        if not hasattr(self.single_sampler, "set_epoch_and_start_iteration"):
            if start_iteration != 0:
                raise NotImplementedError(
                    f"{type(self.single_sampler)} does not support resuming from a nonzero step."
                )
            self.single_sampler.set_epoch(epoch)
        else:
            self.single_sampler.set_epoch_and_start_iteration(epoch, start_iteration)

    def __iter__(self):
        if not self.balance_batches:
            yield from self.batch_sampler
            return

        for batch_idx in self.batch_sampler:
            if self.sizes is None:
                # Unfortunately, we need to load the data to know the image sizes
                data_list = [self.dataset[idx] for idx in batch_idx]

                if self.mode == "atoms":
                    sizes = [data.num_nodes for data in data_list]
                elif self.mode == "neighbors":
                    sizes = [data.edge_index.shape[1] for data in data_list]
                else:
                    raise NotImplementedError(
                        f"Unknown load balancing mode: {self.mode}"
                    )
            else:
                sizes = [self.sizes[idx] for idx in batch_idx]

            idx_sizes = torch.stack([torch.tensor(batch_idx), torch.tensor(sizes)])
            idx_sizes_all = distutils.all_gather(idx_sizes, device=self.device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            if gp_utils.initialized():
                idx_sizes_all = torch.unique(input=idx_sizes_all, dim=1)
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = balanced_partition(
                sizes_all.numpy(), num_parts=self.num_replicas
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[local_idx_balanced[self.rank]]

class OCPDistributedDataParallel(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class OCPDataParallel(torch.nn.DataParallel):
    def __init__(self, module, output_device, num_gpus):
        if num_gpus < 0:
            raise ValueError("# GPUs must be positive.")
        if num_gpus > torch.cuda.device_count():
            raise ValueError("# GPUs specified larger than available")

        self.src_device = torch.device(output_device)

        self.cpu = False
        if num_gpus == 0:
            self.cpu = True
        elif num_gpus == 1:
            device_ids = [self.src_device]
        else:
            if (
                self.src_device.type == "cuda"
                and self.src_device.index >= num_gpus
            ):
                raise ValueError("Main device must be less than # of GPUs")
            device_ids = list(range(num_gpus))

        if self.cpu:
            super(torch.nn.DataParallel, self).__init__()
            self.module = module

        else:
            super(OCPDataParallel, self).__init__(
                module=module,
                device_ids=device_ids,
                output_device=self.src_device,
            )

    def forward(self, batch_list):
        if self.cpu:
            return self.module(batch_list[0])

        if len(self.device_ids) == 1:
            return self.module(batch_list[0].to(f"cuda:{self.device_ids[0]}"))

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    (
                        "Module must have its parameters and buffers on device "
                        "{} but found one of them on device {}."
                    ).format(self.src_device, t.device)
                )

        inputs = [
            batch.to(f"cuda:{self.device_ids[i]}")
            for i, batch in enumerate(batch_list)
        ]
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def extract_features(self, batch_list, main_graph=None):
        import pdb; pdb.set_trace()
        if self.cpu:
            return self.module.extract_features((batch_list[0], main_graph))

        if len(self.device_ids) == 1:
            return self.module.extract_features(
                (batch_list[0].to(f"cuda:{self.device_ids[0]}"), main_graph)
            )

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    (
                        "Module must have its parameters and buffers on device "
                        "{} but found one of them on device {}."
                    ).format(self.src_device, t.device)
                )

        inputs = [
            (batch.to(f"cuda:{self.device_ids[i]}"), main_graph)
            for i, batch in enumerate(batch_list)
        ]
        replicas = self.replicate(
            self.module.extract_features, self.device_ids[: len(inputs)]
        )
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

class ParallelCollater:
    def __init__(self, num_gpus, otf_graph=False):
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph

    def __call__(self, data_list):
        if self.num_gpus in [0, 1]:  # adds cpu-only case
            batch = data_list_collater(data_list, otf_graph=self.otf_graph)
            return [batch]

        else:
            num_devices = min(self.num_gpus, len(data_list))

            count = torch.tensor([data.num_nodes for data in data_list])
            cumsum = count.cumsum(0)
            cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
            device_id = (
                num_devices * cumsum.to(torch.float) / cumsum[-1].item()
            )
            device_id = (device_id[:-1] + device_id[1:]) / 2.0
            device_id = device_id.to(torch.long)
            split = device_id.bincount().cumsum(0)
            split = torch.cat([split.new_zeros(1), split], dim=0)
            split = torch.unique(split, sorted=True)
            split = split.tolist()

            return [
                data_list_collater(data_list[split[i] : split[i + 1]])
                for i in range(len(split) - 1)
            ]

class DistributedWeightedSampler(Sampler):
    """
    Based on these two references
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
    https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#WeightedRandomSampler
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: Sequence[float],  # has same length as dataset
        num_replicas: Optional[int] = None,  # total nr. GPUs
        rank: Optional[int] = None,  # which GPU out of num_replicas
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,  # duplicates a few samples if needed
        replacement: bool = True,  # do not change
    ) -> None:

        assert len(dataset) == len(
            weights
        ), "There has to be a weights for each data point"
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                "weights have shape {}".format(tuple(weights_tensor.shape))
            )

        self.weights = weights_tensor
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        if not isinstance(replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(replacement)
            )
        self.replacement = replacement
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[int]:

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # subsample weights
        weights = self.weights[indices]

        rand_tensor = torch.multinomial(
            input=weights,
            num_samples=len(weights),
            replacement=self.replacement,
            generator=g,
        )

        weighted_sample_indices = torch.tensor(indices)[rand_tensor].tolist()
        # the following line returns the origin labels
        # sample_weights = weights[rand_tensor].tolist()

        return iter(weighted_sample_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch