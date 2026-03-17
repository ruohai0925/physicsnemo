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

import hydra
from omegaconf import DictConfig
from math import ceil
import mlflow

from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

# ===================================================================
# 问题3: How to use Fourier Neural Operator architecture in PhysicsNeMo Sym
# ===================================================================
# FNO模型导入 - 这是PhysicsNeMo中Fourier Neural Operator的核心实现
from physicsnemo.models.fno import FNO

# ===================================================================
# 问题1: How to load grid data and set up data-driven constraints
# ===================================================================
# Darcy2D数据管道 - 负责实时生成网格数据和设置数据驱动的约束
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

# 分布式管理和工具函数
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger

# ===================================================================
# 问题2: How to create a grid validator node
# ===================================================================
# 网格验证器 - 用于验证模型预测结果和可视化
from validator import GridValidator


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def darcy_trainer(cfg: DictConfig) -> None:
    """Training for the 2D Darcy flow benchmark problem.

    This training script demonstrates how to set up a data-driven model for a 2D Darcy flow
    using Fourier Neural Operators (FNO) and acts as a benchmark for this type of operator.
    Training data is generated in-situ via the Darcy2D data loader from PhysicsNeMo. Darcy2D
    continuously generates data previously unseen by the model, i.e. the model is trained
    over a single epoch of a training set consisting of
    (cfg.training.max_pseudo_epochs*cfg.training.pseudo_epoch_sample_size) unique samples.
    Pseudo_epochs were introduced to leverage the LaunchLogger and its MLFlow integration.
    """
    
    # ===================================================================
    # 分布式环境初始化
    # ===================================================================
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # ===================================================================
    # MLFlow实验跟踪设置
    # ===================================================================
    # 设置MLflow跟踪URI - 用于实验管理和可视化
    mlflow.set_tracking_uri("http://localhost:2458")
    mlflow.autolog()  # 自动记录训练指标

    # ===================================================================
    # 日志系统初始化
    # ===================================================================
    # 初始化监控系统
    log = PythonLogger(name="darcy_fno")
    log.file_logging()  # 启用文件日志记录
    LaunchLogger.initialize()  # PhysicsNeMo启动日志记录器

    # ===================================================================
    # 问题3: How to use Fourier Neural Operator architecture in PhysicsNeMo Sym
    # ===================================================================
    # 定义FNO模型 - 这是Fourier Neural Operator的核心架构
    model = FNO(
        # 输入通道数 - 对于Darcy问题，输入是渗透率场，所以是1个通道
        in_channels=cfg.arch.fno.in_channels,  # 1
        
        # 输出通道数 - 输出是压力场，所以是1个通道
        out_channels=cfg.arch.decoder.out_features,  # 1
        
        # 解码器层数和大小 - 用于将FNO特征映射到最终输出
        decoder_layers=cfg.arch.decoder.layers,  # 1
        decoder_layer_size=cfg.arch.decoder.layer_size,  # 32
        
        # FNO核心参数
        dimension=cfg.arch.fno.dimension,  # 2D问题，所以是2
        latent_channels=cfg.arch.fno.latent_channels,  # 32 - 潜在特征维度
        num_fno_layers=cfg.arch.fno.fno_layers,  # 4 - FNO层数
        num_fno_modes=cfg.arch.fno.fno_modes,  # 12 - 保留的Fourier模式数
        padding=cfg.arch.fno.padding,  # 9 - 填充大小
    ).to(dist.device)  # 将模型移动到指定设备（GPU/CPU）
    
    """
    FNO架构说明：
    - FNO通过Fourier变换在频域中进行卷积操作
    - 它能够学习算子（从函数到函数的映射）
    - 对于Darcy问题：渗透率场 → 压力场
    - 优势：能够处理不同分辨率的输入，具有平移不变性
    
    FNO层维度变化详细过程：
    1. 输入: [64, 32, 256, 256] (Lift网络输出)
    2. 填充: [64, 32, 265, 265] (添加9像素填充，减少边界效应)
    3. FNO处理: [64, 32, 265, 265] (4层FNO，每层包含频域卷积和空间卷积)
    4. 移除填充: [64, 32, 256, 256] (恢复原始空间分辨率)
    
    频域处理过程：
    - FFT变换: [64, 32, 265, 265] → [64, 32, 265, 133] (频域)
    - 频域卷积: 只处理前12个和后12个Fourier模式
    - 逆FFT变换: [64, 32, 265, 133] → [64, 32, 265, 265] (空间域)
    """

    # ===================================================================
    # 损失函数和优化器设置
    # ===================================================================
    # 使用均方误差损失函数 - 适合回归问题
    loss_fun = MSELoss(reduction="mean")
    
    # Adam优化器 - 自适应学习率，适合深度学习
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    
    # 学习率调度器 - 指数衰减
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )

    # ===================================================================
    # 问题1: How to load grid data and set up data-driven constraints
    # ===================================================================
    # 数据归一化设置 - 这是数据驱动约束的重要组成部分
    norm_vars = cfg.normaliser
    normaliser = {
        # 渗透率场的归一化参数（均值，标准差）
        "permeability": (norm_vars.permeability.mean, norm_vars.permeability.std_dev),
        # Darcy压力场的归一化参数（均值，标准差）
        "darcy": (norm_vars.darcy.mean, norm_vars.darcy.std_dev),
    }
    
    """
    数据归一化说明：
    - 归一化是数据驱动约束的关键部分
    - 确保输入和输出数据在相似的数值范围内
    - 提高训练稳定性和收敛速度
    - 对于Darcy问题：
      * 渗透率：均值1.25，标准差0.75
      * 压力：均值4.52E-2，标准差2.79E-2
    """
    
    # 创建Darcy2D数据加载器 - 这是网格数据加载的核心
    dataloader = Darcy2D(
        resolution=cfg.training.resolution,  # 256x256网格分辨率
        batch_size=cfg.training.batch_size,  # 64个样本/批次
        normaliser=normaliser,  # 应用归一化
    )
    
    """
    Darcy2D数据加载器说明：
    - 实时生成数据：不需要预存储大量数据
    - 基于随机Fourier级数生成渗透率场
    - 使用GPU加速的多网格Jacobi迭代求解Darcy方程
    - 自动处理边界条件和物理约束
    - 支持批量生成，提高训练效率
    
    数据生成流程：
    1. 生成随机Fourier系数
    2. 通过逆Fourier变换生成渗透率场
    3. 应用阈值处理得到分段常数函数
    4. 使用多网格方法求解Darcy方程得到压力场
    5. 应用归一化处理
    """

    # ===================================================================
    # 问题2: How to create a grid validator node
    # ===================================================================
    # 创建网格验证器 - 用于验证和可视化模型性能
    validator = GridValidator(loss_fun=MSELoss(reduction="mean"))
    
    """
    GridValidator说明：
    - 比较模型预测和真实值
    - 反归一化数据用于可视化
    - 生成验证图像和误差分析
    - 支持MLFlow集成，自动记录验证结果
    - 提供相对误差计算和可视化
    """

    # ===================================================================
    # 检查点管理设置
    # ===================================================================
    ckpt_args = {
        "path": f"./checkpoints",  # 检查点保存路径
        "optimizer": optimizer,    # 优化器状态
        "scheduler": scheduler,    # 学习率调度器状态
        "models": model,          # 模型权重
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # ===================================================================
    # 训练参数计算
    # ===================================================================
    # 计算每个伪epoch的步数
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    
    # 日志参数设置
    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    
    # 检查批次大小兼容性
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of "
            f"batch size: {steps_per_pseudo_epoch * cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of "
            f"batch size: {validation_iters * cfg.training.batch_size}"
        )

    # ===================================================================
    # 前向传播函数定义
    # ===================================================================
    # 训练时的前向传播 - 使用StaticCaptureTraining进行优化
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        """
        训练前向传播函数
        
        参数：
        - invars: 输入变量（渗透率场）[64, 1, 256, 256]
        - target: 目标变量（真实压力场）[64, 1, 256, 256]
        
        返回：
        - loss: 损失值 (scalar)
        
        维度变化：
        - invars: [64, 1, 256, 256] → FNO处理 → pred: [64, 1, 256, 256]
        - target: [64, 1, 256, 256]
        - loss: scalar (MSE)
        """
        pred = model(invars)  # FNO模型前向传播 [64, 1, 256, 256]
        loss = loss_fun(pred, target)  # 计算MSE损失 (scalar)
        return loss

    # 评估时的前向传播 - 使用StaticCaptureEvaluateNoGrad进行优化
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        """
        评估前向传播函数
        
        参数：
        - invars: 输入变量（渗透率场）[64, 1, 256, 256]
        
        返回：
        - pred: 模型预测（压力场）[64, 1, 256, 256]
        
        维度变化：
        - invars: [64, 1, 256, 256] → FNO处理 → pred: [64, 1, 256, 256]
        """
        return model(invars)

    # ===================================================================
    # 训练循环
    # ===================================================================
    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")

    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # 使用LaunchLogger包装epoch，用于控制台/MLFlow日志
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            # 训练循环
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                # 从数据加载器获取批次数据
                # batch["permeability"]: 渗透率场 [batch_size, 1, 256, 256]
                # batch["darcy"]: 压力场 [batch_size, 1, 256, 256]
                
                # 维度分析（可选：取消注释以查看实际维度）
                # print(f"Input permeability shape: {batch['permeability'].shape}")  # [64, 1, 256, 256]
                # print(f"Target darcy shape: {batch['darcy'].shape}")              # [64, 1, 256, 256]
                
                loss = forward_train(batch["permeability"], batch["darcy"])
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # 保存检查点
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # ===================================================================
        # 验证步骤 - 使用GridValidator进行验证
        # ===================================================================
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                for _, batch in zip(range(validation_iters), dataloader):
                    # 使用GridValidator进行验证
                    val_loss = validator.compare(
                        batch["permeability"],  # 输入渗透率场
                        batch["darcy"],         # 真实压力场
                        forward_eval(batch["permeability"]),  # 模型预测
                        pseudo_epoch,          # 当前epoch
                        logger,                # 日志记录器
                    )
                    total_loss += val_loss
                logger.log_epoch({"Validation error": total_loss / validation_iters})

        # 更新学习率
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    # 保存最终检查点
    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")


if __name__ == "__main__":
    darcy_trainer()
