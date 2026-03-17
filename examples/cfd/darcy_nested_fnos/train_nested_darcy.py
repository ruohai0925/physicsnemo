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

# ===================================================================
# 导入必要的库和模块
# ===================================================================
import os
import glob
import hydra
from typing import Tuple
from omegaconf import DictConfig
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    LaunchLogger,
)
from physicsnemo.utils.logging.mlflow import initialize_mlflow
from utils import NestedDarcyDataset, GridValidator

"""
Nested FNO训练脚本说明：
- 这是嵌套Fourier Neural Operator的训练脚本
- 支持多级模型训练（ref0: 全局模型, ref1: 局部细化模型）
- 使用预生成的多分辨率数据集
- 支持分布式训练和MLFlow监控
"""


def InitializeLoggers(cfg: DictConfig) -> Tuple[DistributedManager, PythonLogger]:
    """
    初始化分布式管理和日志系统
    
    这个函数设置训练所需的基础设施，包括分布式训练环境和日志记录系统。
    
    Parameters
    ----------
    cfg : DictConfig
        配置文件参数，必须包含model字段指定要训练的模型
        
    Returns
    -------
    Tuple[DistributedManager, PythonLogger]
        分布式管理器和日志记录器的元组
        
    注意：
    - 必须通过命令行参数指定模型：+model=ref0 或 +model=ref1
    - 支持多GPU分布式训练
    - 使用MLFlow进行实验跟踪
    """
    # 初始化分布式管理器 - 整个脚本中只调用一次！
    DistributedManager.initialize()
    dist = DistributedManager()  # 在其他地方需要时调用
    logger = PythonLogger(name="darcy_nested_fno")

    # 检查是否指定了要训练的模型
    assert hasattr(cfg, "model"), logger.error(
        f"必须指定要训练的模型: $ python {__file__.split(os.sep)[-1]} +model=<model_name>"
    )
    logger.info(f"开始训练模型 {cfg.model}")

    # 初始化MLFlow监控系统
    initialize_mlflow(
        experiment_name=f"Nested FNO, model: {cfg.model}",
        experiment_desc=f"训练模型 {cfg.model} 用于嵌套FNO",
        run_name=f"Nested FNO training, model: {cfg.model}",
        run_desc=f"训练模型 {cfg.model} 用于嵌套FNO",
        user_name="Gretchen Ross",
        mode="offline",  # 离线模式，适合远程训练
    )
    LaunchLogger.initialize(use_mlflow=True)  # PhysicsNeMo启动日志器

    return dist, RankZeroLoggingWrapper(logger, dist)


class SetUpInfrastructure:
    """
    训练基础设施设置类
    
    这个类包含了训练所需的所有重要对象，包括模型、数据加载器、优化器等。
    根据指定的模型级别（ref0或ref1）设置相应的配置。
    
    Parameters
    ----------
    cfg : DictConfig
        配置文件参数
    dist : DistributedManager
        分布式管理器实例，存储并行环境信息
    logger : PythonLogger
        命令行输出日志器
    """

    def __init__(
        self, cfg: DictConfig, dist: DistributedManager, logger: PythonLogger
    ) -> None:
        # ===================================================================
        # 1. 模型配置和损失函数设置
        # ===================================================================
        # 从模型名称中提取级别（ref0 -> 0, ref1 -> 1）
        level = int(cfg.model[-1])
        model_cfg = cfg.arch[cfg.model]
        
        # 定义损失函数 - 使用均方误差
        loss_fun = MSELoss(reduction="mean")
        
        # 定义归一化参数
        norm = {
            "permeability": (
                cfg.normaliser.permeability.mean,    # 渗透率均值
                cfg.normaliser.permeability.std,     # 渗透率标准差
            ),
            "darcy": (
                cfg.normaliser.darcy.mean,           # Darcy压力均值
                cfg.normaliser.darcy.std,            # Darcy压力标准差
            ),
        }
        
        """
        归一化参数说明：
        - 渗透率：均值1.25，标准差0.75
        - Darcy压力：均值2.6E-2，标准差1.5E-2
        - 这些参数用于数据预处理和后处理
        """

        # ===================================================================
        # 2. 数据加载器设置
        # ===================================================================
        # 创建训练数据集
        self.training_set = NestedDarcyDataset(
            mode="train",                              # 训练模式
            data_path=cfg.training.training_set,      # 训练数据路径
            model_name=cfg.model,                     # 模型名称（ref0或ref1）
            norm=norm,                                # 归一化参数
            log=logger,                               # 日志器
        )
        
        # 创建验证数据集
        self.valid_set = NestedDarcyDataset(
            mode="train",                              # 使用训练模式加载验证数据
            data_path=cfg.validation.validation_set,  # 验证数据路径
            model_name=cfg.model,                     # 模型名称（ref0或ref1）
            norm=norm,                                # 归一化参数
            log=logger,                               # 日志器
        )

        # 记录数据集大小信息
        logger.log(
            f"训练集包含 {len(self.training_set)} 个样本, "
            + f"验证集包含 {len(self.valid_set)} 个样本."
        )
        
        """
        NestedDarcyDataset说明：
        - 根据模型级别加载相应的数据
        - ref0: 加载全局粗分辨率数据（256×256）
        - ref1: 加载局部细化数据（128×128）+ 父级预测
        - 自动处理数据归一化和设备放置
        """

        # ===================================================================
        # 3. 分布式采样器和数据加载器设置
        # ===================================================================
        # 创建分布式训练采样器
        train_sampler = DistributedSampler(
            self.training_set,
            num_replicas=dist.world_size,    # 总进程数
            rank=dist.local_rank,           # 当前进程排名
            shuffle=True,                   # 是否打乱数据
            drop_last=False,                # 是否丢弃最后不完整的批次
        )

        # 创建分布式验证采样器
        valid_sampler = DistributedSampler(
            self.valid_set,
            num_replicas=dist.world_size,    # 总进程数
            rank=dist.local_rank,           # 当前进程排名
            shuffle=True,                   # 是否打乱数据
            drop_last=False,                # 是否丢弃最后不完整的批次
        )

        # 创建训练数据加载器
        self.train_loader = DataLoader(
            self.training_set,
            batch_size=cfg.training.batch_size,  # 批次大小
            shuffle=False,                       # 使用采样器，不需要额外打乱
            sampler=train_sampler,               # 分布式采样器
        )
        
        # 创建验证数据加载器
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=cfg.validation.batch_size,  # 验证批次大小
            shuffle=False,                         # 使用采样器，不需要额外打乱
            sampler=valid_sampler,                 # 分布式采样器
        )
        
        # 创建网格验证器
        self.validator = GridValidator(loss_fun=loss_fun, norm=norm)
        
        """
        分布式训练说明：
        - 使用DistributedSampler确保每个GPU处理不同的数据子集
        - 支持多GPU和多节点训练
        - 自动处理数据分布和同步
        """
        # ===================================================================
        # 4. FNO模型创建和配置
        # ===================================================================
        self.model = FNO(
            in_channels=model_cfg.fno.in_channels,      # 输入通道数
            out_channels=model_cfg.decoder.out_features, # 输出通道数
            decoder_layers=model_cfg.decoder.layers,     # 解码器层数
            decoder_layer_size=model_cfg.decoder.layer_size, # 解码器层大小
            dimension=model_cfg.fno.dimension,           # 问题维度（2D）
            latent_channels=model_cfg.fno.latent_channels, # 潜在通道数
            num_fno_layers=model_cfg.fno.fno_layers,     # FNO层数
            num_fno_modes=model_cfg.fno.fno_modes,       # Fourier模式数
            padding=model_cfg.fno.padding,               # 填充大小
        ).to(dist.device)  # 移动到指定设备（GPU/CPU）
        
        """
        FNO模型配置说明：
        - ref0模型：in_channels=1（只有渗透率输入）
        - ref1模型：in_channels=2（渗透率 + 父级预测）
        - 其他参数在两个模型中相同
        - 使用相同的FNO架构，但输入通道数不同
        """

        # ===================================================================
        # 5. 分布式数据并行设置
        # ===================================================================
        # 如果使用多GPU训练，包装模型为分布式数据并行
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],           # 设备ID列表
                output_device=dist.device,              # 输出设备
                broadcast_buffers=dist.broadcast_buffers, # 是否广播缓冲区
                find_unused_parameters=dist.find_unused_parameters, # 是否查找未使用参数
            )

        # ===================================================================
        # 6. 优化器和学习率调度器设置
        # ===================================================================
        # 创建Adam优化器
        self.optimizer = Adam(self.model.parameters(), lr=cfg.scheduler.initial_lr)
        
        # 创建学习率调度器（指数衰减）
        self.scheduler = lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda step: cfg.scheduler.decay_rate**step  # 指数衰减函数
        )
        
        """
        优化器配置说明：
        - 使用Adam优化器，学习率1.E-3
        - 学习率每2个epoch衰减0.95倍
        - 支持分布式训练的梯度同步
        """
        # ===================================================================
        # 7. 日志和检查点配置
        # ===================================================================
        # 日志记录参数
        self.log_args = {
            "name_space": "train",                    # 日志命名空间
            "num_mini_batch": len(self.train_loader), # 每个epoch的mini-batch数量
            "epoch_alert_freq": 1,                    # epoch提醒频率
        }
        
        # 常规检查点保存参数
        self.ckpt_args = {
            "path": f"./checkpoints/all/{cfg.model}",  # 检查点保存路径
            "optimizer": self.optimizer,               # 优化器状态
            "scheduler": self.scheduler,               # 调度器状态
            "models": self.model,                      # 模型权重
        }
        
        # 最佳检查点保存参数
        self.bst_ckpt_args = {
            "path": f"./checkpoints/best/{cfg.model}", # 最佳检查点保存路径
            "optimizer": self.optimizer,               # 优化器状态
            "scheduler": self.scheduler,               # 调度器状态
            "models": self.model,                      # 模型权重
        }
        
        """
        检查点管理说明：
        - 每5个epoch保存一次常规检查点
        - 当验证损失改善时保存最佳检查点
        - 支持训练中断后恢复
        """

        # ===================================================================
        # 8. 前向传播函数定义
        # ===================================================================
        # 定义训练前向传播函数
        @StaticCaptureTraining(
            model=self.model,
            optim=self.optimizer,
            logger=logger,
            use_amp=False,      # 不使用自动混合精度
            use_graphs=False,   # 不使用CUDA图
        )
        def _forward_train(invars, target):
            """
            训练前向传播函数
            
            参数：
            - invars: 输入变量（渗透率场或渗透率+父级预测）
            - target: 目标变量（真实压力场）
            
            返回：
            - loss: 损失值 (scalar)
            
            维度变化：
            - ref0: invars [64, 1, 256, 256] → pred [64, 1, 256, 256]
            - ref1: invars [64, 2, 128, 128] → pred [64, 1, 128, 128]
            """
            pred = self.model(invars)        # FNO模型前向传播
            loss = loss_fun(pred, target)    # 计算MSE损失
            return loss

        # 定义评估前向传播函数
        @StaticCaptureEvaluateNoGrad(
            model=self.model, 
            logger=logger, 
            use_amp=False,      # 不使用自动混合精度
            use_graphs=False    # 不使用CUDA图
        )
        def _forward_eval(invars):
            """
            评估前向传播函数（无梯度计算）
            
            参数：
            - invars: 输入变量
            
            返回：
            - prediction: 模型预测结果
            """
            return self.model(invars)

        # 将函数绑定到实例
        self.forward_train = _forward_train
        self.forward_eval = _forward_eval


def TrainModel(cfg: DictConfig, base: SetUpInfrastructure, loaded_epoch: int) -> None:
    """
    训练循环函数
    
    执行完整的训练过程，包括前向传播、反向传播、验证和检查点保存。
    
    Parameters
    ----------
    cfg : DictConfig
        配置文件参数
    base : SetUpInfrastructure
        训练基础设施对象，包含模型、数据加载器等
    loaded_epoch : int
        训练重启的起始epoch，==0表示从头开始训练
    """

    # 初始化最小验证损失，用于保存最佳模型
    min_valid_loss = 9.0e9
    
    # 训练循环：从loaded_epoch+1开始到最大epoch
    for epoch in range(max(1, loaded_epoch + 1), cfg.training.max_epochs + 1):
        # ===================================================================
        # 训练阶段
        # ===================================================================
        # 使用LaunchLogger包装epoch，用于控制台和MLFlow日志记录
        with LaunchLogger(**base.log_args, epoch=epoch) as log:
            # 遍历训练数据加载器
            for batch in base.train_loader:
                # 执行前向传播和反向传播
                loss = base.forward_train(batch["permeability"], batch["darcy"])
                # 记录mini-batch损失
                log.log_minibatch({"loss": loss.detach()})
            # 记录epoch级别的学习率
            log.log_epoch({"Learning Rate": base.optimizer.param_groups[0]["lr"]})

        # ===================================================================
        # 验证阶段
        # ===================================================================
        # 验证条件：每5个epoch验证一次，或记录结果时验证，或最后一个epoch
        if (
            epoch % cfg.validation.validation_epochs == 0
            or epoch % cfg.training.rec_results_freq == 0
            or epoch == cfg.training.max_epochs
        ):
            with LaunchLogger("valid", epoch=epoch) as log:
                total_loss = 0.0
                # 遍历验证数据加载器
                for batch in base.valid_loader:
                    # 执行验证：比较预测和真实值
                    loss = base.validator.compare(
                        batch["permeability"],                    # 输入渗透率
                        batch["darcy"],                          # 真实压力场
                        base.forward_eval(batch["permeability"]), # 模型预测
                        epoch,                                   # 当前epoch
                        log,                                     # 日志器
                    )
                    # 累积验证损失（按样本数量加权）
                    total_loss += loss * batch["darcy"].shape[0] / len(base.valid_set)
                # 记录验证误差
                log.log_epoch({"Validation error": total_loss})

        # ===================================================================
        # 检查点保存
        # ===================================================================
        # 保存条件：每5个epoch保存一次，或最后一个epoch
        if (
            epoch % cfg.training.rec_results_freq == 0
            or epoch == cfg.training.max_epochs
        ):
            # 保存常规检查点
            save_checkpoint(**base.ckpt_args, epoch=epoch)
            
            # 如果当前验证损失是最佳的，保存最佳检查点
            if total_loss < min_valid_loss:
                min_valid_loss = total_loss
                # 删除之前的最佳检查点
                for ckpt in glob.glob(base.bst_ckpt_args["path"] + "/*.pt"):
                    os.remove(ckpt)
                # 保存新的最佳检查点
                save_checkpoint(**base.bst_ckpt_args, epoch=epoch)

        # ===================================================================
        # 学习率更新
        # ===================================================================
        # 每2个epoch更新一次学习率
        if epoch % cfg.scheduler.decay_epochs == 0:
            base.scheduler.step()


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def nested_darcy_trainer(cfg: DictConfig) -> None:
    """
    嵌套2D Darcy流问题的主训练函数
    
    这个训练脚本演示了如何使用嵌套Fourier Neural Operators (nFNO) 
    为嵌套2D Darcy流设置数据驱动模型。
    
    nFNOs本质上是多个独立FNO模型的级联。各个FNO可以独立训练，
    训练顺序不重要。顺序只在微调（待实现）和推理时变得重要。
    
    使用方法：
    - 训练ref0模型：python train_nested_darcy.py +model=ref0
    - 训练ref1模型：python train_nested_darcy.py +model=ref1
    - 多GPU训练：mpirun -n 2 python train_nested_darcy.py +model=ref0
    """

    # ===================================================================
    # 1. 初始化日志系统
    # ===================================================================
    dist, logger = InitializeLoggers(cfg)

    # ===================================================================
    # 2. 设置训练基础设施
    # ===================================================================
    base = SetUpInfrastructure(cfg, dist, logger)

    # ===================================================================
    # 3. 检查是否有检查点可以恢复
    # ===================================================================
    loaded_epoch = load_checkpoint(**base.ckpt_args, device=dist.device)
    if loaded_epoch == 0:
        logger.success("开始训练...")
    else:
        logger.warning(f"从第 {loaded_epoch + 1} 个epoch恢复训练.")

    # ===================================================================
    # 4. 执行训练
    # ===================================================================
    TrainModel(cfg, base, loaded_epoch)
    logger.success("训练完成 *yay*")


if __name__ == "__main__":
    nested_darcy_trainer()
