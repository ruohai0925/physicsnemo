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
Darcy流问题的Transolver训练脚本

本脚本实现了使用Transolver模型求解2D Darcy流问题的完整训练流程。
主要功能包括：
1. 模型初始化和配置
2. 数据加载和预处理
3. 训练循环和验证
4. 模型保存和日志记录

训练过程：
- 输入：渗透率场 k(x,y)
- 输出：压力场 p(x,y)
- 目标：学习从渗透率到压力场的映射关系

作者：Haixu Wu, Huakun Luo, Haowen Wang
"""

import hydra
from omegaconf import DictConfig
from math import ceil

from torch.nn import MSELoss
from utils.testloss import TestLoss
from torch.optim import Adam, lr_scheduler

from physicsnemo.models.transolver import Transolver
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger
from physicsnemo.utils.logging.mlflow import initialize_mlflow

from validator import GridValidator
from einops import rearrange


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def darcy_trainer(cfg: DictConfig) -> None:
    """
    2D Darcy流问题的Transolver训练主函数
    
    参数:
        cfg (DictConfig): Hydra配置对象，包含所有训练参数
        
    训练流程:
        1. 初始化分布式训练环境
        2. 设置日志和监控系统
        3. 创建Transolver模型
        4. 配置优化器和学习率调度器
        5. 创建数据加载器
        6. 执行训练循环
        7. 定期验证和保存模型
    """
    # 初始化分布式训练管理器（整个脚本中只调用一次）
    DistributedManager.initialize()
    dist = DistributedManager()  # 获取分布式管理器实例

    # 初始化日志系统
    log = PythonLogger(name="darcy_transolver")
    log.file_logging()  # 启用文件日志记录
    
    # 初始化MLFlow实验跟踪
    initialize_mlflow(
        experiment_name=f"Darcy_Transolver",
        experiment_desc=f"training a Transformer-based PDE solver for the Darcy problem",
        run_name=f"Darcy Transolver training",
        run_desc=f"training Transolver for Darcy",
        user_name="Haixu Wu, Huakun Luo, Haowen Wang",
        mode="offline",  # 离线模式，避免网络依赖
    )
    LaunchLogger.initialize(use_mlflow=True)  # 初始化PhysicsNeMo启动日志器

    # ==================== 模型定义 ====================
    # 创建Transolver模型实例
    model = Transolver(
        out_dim=cfg.model.out_dim,                    # 输出维度：压力场通道数
        embedding_dim=cfg.model.embedding_dim,        # 嵌入维度：位置编码维度
        n_layers=cfg.model.n_layers,                  # Transformer层数
        n_hidden=cfg.model.n_hidden,                  # 隐藏层维度
        dropout=cfg.model.dropout,                    # Dropout率
        n_head=cfg.model.n_head,                      # 注意力头数
        act=cfg.model.act,                            # 激活函数类型
        mlp_ratio=cfg.model.mlp_ratio,                # MLP扩展比例
        functional_dim=cfg.model.functional_dim,      # 功能维度：输入特征维度
        slice_num=cfg.model.slice_num,                # 注意力切片数量
        unified_pos=True,                             # 使用统一位置编码
        ref=cfg.model.ref,                            # 位置编码参考维度
        structured_shape=[cfg.data.resolution, cfg.data.resolution],  # 结构化网格形状
        use_te=cfg.model.use_te,                      # 是否使用Transformer Engine
        time_input=cfg.model.time_input,              # 是否包含时间输入
    ).to(dist.device)  # 将模型移动到指定设备（GPU/CPU）

    # ==================== 损失函数和优化器 ====================
    # 使用相对L2损失函数（TestLoss）
    loss_fun = TestLoss(size_average=False)
    
    # Adam优化器
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    
    # 学习率调度器：指数衰减
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )
    
    # ==================== 数据归一化器 ====================
    # 从配置文件读取归一化参数
    norm_vars = cfg.normaliser
    normaliser = {
        "permeability": (norm_vars.permeability.mean, norm_vars.permeability.std_dev),  # 渗透率归一化参数
        "darcy": (norm_vars.darcy.mean, norm_vars.darcy.std_dev),                      # Darcy解归一化参数
    }
    
    # ==================== 数据加载器 ====================
    # 创建Darcy2D数据管道（在线生成数据）
    dataloader = Darcy2D(
        resolution=cfg.training.resolution,    # 网格分辨率
        batch_size=cfg.training.batch_size,    # 批次大小
        normaliser=normaliser,                 # 归一化器
    )
    
    # ==================== 验证器 ====================
    # 创建网格验证器，用于验证和可视化
    validator = GridValidator(loss_fun=TestLoss(size_average=False), norm=normaliser)

    # ==================== 检查点管理 ====================
    # 设置检查点保存参数
    ckpt_args = {
        "path": f"./checkpoints",      # 检查点保存路径
        "optimizer": optimizer,        # 优化器状态
        "scheduler": scheduler,        # 学习率调度器状态
        "models": model,               # 模型参数
    }
    # 尝试加载之前的检查点，返回已训练的伪epoch数
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # ==================== 训练参数计算 ====================
    # 计算每个伪epoch的步数（每个伪epoch处理的样本数 / 批次大小）
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    # 计算验证迭代次数
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)
    
    # 日志记录参数
    log_args = {
        "name_space": "train",                    # 日志命名空间
        "num_mini_batch": steps_per_pseudo_epoch, # 每个epoch的mini-batch数量
        "epoch_alert_freq": 1,                    # epoch提醒频率
    }
    
    # 检查并调整批次大小对齐
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

    # ==================== 前向传播函数定义 ====================
    # 训练时的前向传播函数（包含梯度计算）
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        """
        训练前向传播函数
        
        参数:
            invars: 输入变量（渗透率场），形状 [batch, channels, height, width]
            target: 目标变量（压力场），形状 [batch, channels, height, width]
            
        返回:
            loss: 计算得到的损失值
        """
        invars_shape = invars.shape  # 保存原始形状用于调试
        # 将4D张量重塑为3D：[batch, channels, height, width] -> [batch, height*width, channels]
        # 这是为了适配Transformer的序列输入格式
        invars = rearrange(invars, "b c h w -> b (h w) c")
        # 通过Transolver模型进行前向传播
        pred = model(invars)
        # 计算损失（相对L2误差）
        loss = loss_fun(pred, target)
        return loss

    # 评估时的前向传播函数（不计算梯度）
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        """
        评估前向传播函数（无梯度计算）
        
        参数:
            invars: 输入变量（渗透率场），形状 [batch, channels, height, width]
            
        返回:
            model(invars): 模型预测结果
        """
        # 将4D张量重塑为3D：[batch, channels, height, width] -> [batch, height*width, channels]
        # 这是为了适配Transformer的序列输入格式，与forward_train保持一致
        invars = rearrange(invars, "b c h w -> b (h w) c")
        return model(invars)

    # ==================== 训练循环 ====================
    # 检查是否从检查点恢复训练
    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")

    # 主训练循环：遍历所有伪epoch
    for pseudo_epoch in range(
        max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
    ):
        # 使用LaunchLogger包装epoch，用于控制台和MLFlow日志记录
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            # 训练阶段：遍历当前epoch的所有批次
            for _, batch in zip(range(steps_per_pseudo_epoch), dataloader):
                # 执行前向传播和反向传播
                # batch["permeability"]: 输入渗透率场
                # batch["darcy"]: 目标压力场
                loss = forward_train(batch["permeability"], batch["darcy"])
                # 记录mini-batch损失
                logger.log_minibatch({"loss": loss.detach()})
            # 记录epoch级别的学习率
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # 保存检查点（定期保存）
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # 验证阶段（定期验证）
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                # 遍历验证集的所有批次
                for _, batch in zip(range(validation_iters), dataloader):
                    # 执行验证：比较预测结果和真实值
                    val_loss = validator.compare(
                        batch["permeability"],           # 输入渗透率
                        batch["darcy"],                 # 真实压力场
                        forward_eval(batch["permeability"]),  # 模型预测
                        pseudo_epoch,                   # 当前epoch
                        logger,                         # 日志器
                    )
                    total_loss += val_loss
                # 记录平均验证误差
                logger.log_epoch({"Validation error": total_loss / validation_iters})

        # 更新学习率（定期衰减）
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    # 训练完成，保存最终检查点
    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")


if __name__ == "__main__":
    darcy_trainer()
