# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
Darcy流问题的固定数据集数据管道

本模块实现了Darcy2D_fix数据管道，用于加载预生成的Darcy流数据集。
与在线生成数据的Darcy2D不同，本数据管道从预保存的.mat或.npz文件中加载数据。

主要功能：
1. 从文件加载预生成的渗透率和压力场数据
2. 数据归一化和预处理
3. 支持下采样到不同分辨率
4. 提供训练和测试数据迭代器

数据格式：
- 输入：渗透率场 k(x,y) - 形状 [batch, height, width]
- 输出：压力场 p(x,y) - 形状 [batch, height, width]
- 位置：网格坐标 (x,y) - 形状 [height*width, 2]

作者：PhysicsNeMo团队
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import torch
import scipy.io as scio
import torch.distributed as dist

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor

from physicsnemo.utils.profiling import profile


class UnitTransformer:
    """
    数据归一化和反归一化转换器
    
    用于对数据进行标准化处理，将数据转换为均值为0、标准差为1的分布。
    支持编码（归一化）和解码（反归一化）操作。
    """

    def __init__(self, X):
        """
        初始化归一化器
        
        参数:
            X: 输入数据张量，用于计算均值和标准差
        """
        # 计算数据的均值和标准差（在batch和空间维度上）
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8  # 添加小值避免除零

    def to(self, device):
        """将归一化器参数移动到指定设备"""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        """将归一化器参数移动到CUDA设备"""
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        """将归一化器参数移动到CPU设备"""
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        """
        编码（归一化）数据
        
        参数:
            x: 输入数据
            
        返回:
            归一化后的数据：(x - mean) / std
        """
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        """
        解码（反归一化）数据
        
        参数:
            x: 归一化的数据
            
        返回:
            反归一化后的数据：x * std + mean
        """
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component="all"):
        """
        通用转换函数
        
        参数:
            X: 输入数据
            inverse: 是否进行反归一化（True）或归一化（False）
            component: 要处理的组件（"all"表示所有组件）
            
        返回:
            转换后的数据
        """
        if component == "all" or "all-reduce":
            if inverse:
                # 反归一化
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                # 归一化
                return (X - self.mean) / self.std
        else:
            # 处理特定组件
            if inverse:
                orig_shape = X.shape
                return (
                    X * (self.std[:, component] - 1e-8) + self.mean[:, component]
                ).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "Darcy2D"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = False


class Darcy2D_fix(Datapipe):
    """2D Darcy flow benchmark problem datapipe.

    This datapipe continuously generates solutions to the 2D Darcy equation with variable
    permeability. All samples are generated on the fly and is meant to be a benchmark
    problem for testing data driven models. Permeability is drawn from a random Fourier
    series and threshold it to give a piecewise constant function. The solution is obtained
    using a GPU enabled multi-grid Jacobi iterative method.

    Parameters
    ----------
    resolution : int, optional
        Resolution to run simulation at, by default 256
    batch_size : int, optional
        Batch size of simulations, by default 64
    nr_permeability_freq : int, optional
        Number of frequencies to use for generating random permeability. Higher values
        will give higher freq permeability fields., by default 5
    max_permeability : float, optional
        Max permeability, by default 2.0
    min_permeability : float, optional
        Min permeability, by default 0.5
    max_iterations : int, optional
        Maximum iterations to use for each multi-grid, by default 30000
    convergence_threshold : float, optional
        Solver L-Infinity convergence threshold, by default 1e-6
    iterations_per_convergence_check : int, optional
        Number of Jacobi iterations to run before checking convergence, by default 1000
    nr_multigrids : int, optional
        Number of multi-grid levels, by default 4
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys `permeability` and `darcy`. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"

    Raises
    ------
    ValueError
        Incompatable multi-grid and resolution settings
    """

    @profile
    def __init__(
        self,
        resolution: int = 256,
        batch_size: int = 64,
        nr_permeability_freq: int = 5,
        max_permeability: float = 2.0,
        min_permeability: float = 0.5,
        max_iterations: int = 30000,
        convergence_threshold: float = 1e-6,
        iterations_per_convergence_check: int = 1000,
        # nr_multigrids: int = 4,
        # normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        device: Union[str, torch.device] = "cuda",
        train_path: str = None,
        is_test: bool = False,
        x_normalizer: UnitTransformer = None,
        y_normalizer: UnitTransformer = None,
        downsample: int = 5,
    ):
        super().__init__(meta=MetaData())

        # simulation params
        self.resolution = resolution
        self.batch_size = batch_size
        self.nr_permeability_freq = nr_permeability_freq
        self.max_permeability = max_permeability
        self.min_permeability = min_permeability
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.iterations_per_convergence_check = iterations_per_convergence_check

        # Set up device for warp, warp has same naming convention as torch.
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device

        # spatial dims
        self.dx = 1.0 / (self.resolution + 1)  # pad edges by 1 for multi-grid
        self.dim = (self.batch_size, self.resolution + 1, self.resolution + 1)

        self.train_path = train_path
        self.native_resolution = 421  # Native grid size

        # Calculate downsampling factor
        if (self.native_resolution - 1) % (self.resolution - 1) != 0:
            raise ValueError(
                f"Resolution {self.resolution} is not achievable by strided sampling from native resolution {self.native_resolution}."
            )
        self.r = (self.native_resolution - 1) // (self.resolution - 1)
        self.s = self.resolution
        self.dx = 1.0 / self.s
        # Output tenors
        self.output_k = None
        self.output_p = None

        self.is_test = is_test

        if not self.is_test:
            self.n_train = 1024
        else:
            self.n_train = 200

        if self.train_path is not None:
            self.__get_data__()

        if not self.is_test:
            self.x_normalizer = UnitTransformer(self.x_train)
            self.y_normalizer = UnitTransformer(self.y_train)

            self.x_train = self.x_normalizer.encode(self.x_train)
            self.y_train = self.y_normalizer.encode(self.y_train)
        else:
            self.x_train = x_normalizer.encode(self.x_train)
            self.y_train = y_normalizer.encode(self.y_train)

    @profile
    def __get_normalizer__(self):
        return self.x_normalizer, self.y_normalizer

    @profile
    def __get_data__(self):
        if self.train_path.endswith(".mat"):
            data_dict = scio.loadmat(self.train_path)
        elif self.train_path.endswith(".npz"):
            data_dict = np.load(self.train_path)

        # Extract data from dicts:
        self.x_train = data_dict["coeff"]
        self.y_train = data_dict["sol"]

        x = np.linspace(0, 1, self.s)
        y = np.linspace(0, 1, self.s)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).cuda()

        # Downsampling logic
        if self.r > 1:
            # Downsample by slicing
            self.x_train = self.x_train[: self.n_train, :: self.r, :: self.r][
                :, : self.s, : self.s
            ]
            self.y_train = self.y_train[: self.n_train, :: self.r, :: self.r][
                :, : self.s, : self.s
            ]
        else:
            # No downsampling, use full resolution
            self.x_train = self.x_train[: self.n_train, : self.s, : self.s]
            self.y_train = self.y_train[: self.n_train, : self.s, : self.s]

        # Flatten them:
        self.x_train = self.x_train.reshape(self.n_train, -1)
        self.y_train = self.y_train.reshape(self.n_train, -1)

        self.x_train = torch.from_numpy(self.x_train).float().cuda()
        self.y_train = torch.from_numpy(self.y_train).float().cuda()
        # Why are we repeating the postion?
        # print(f"pos shape: {pos.shape}")
        self.pos_train = pos
        self.pos_train_batched = pos.repeat(self.batch_size, 1, 1).cuda()
        # print(f"pos shape post repeat: {self.pos_train.shape}")
        # self.pos_train = pos

    @profile
    def __iter__(self):
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor]]
            Infinite iterator that returns a batch of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution]
        """

        while True:
            # Sample batch_size indices from this rank's shard
            idx = np.random.choice(self.n_train, self.batch_size)
            # All tensors are already on GPU, so no .cuda() needed
            x = self.x_train[idx]
            y = self.y_train[idx]
            yield self.pos_train_batched, x, y

    def __getitem__(self, idx):
        return self.pos_train, self.x_train[idx], self.y_train[idx]

    def __len__(self):
        return self.n_train
