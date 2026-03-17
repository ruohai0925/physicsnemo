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
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils import HDF5MapStyleDataset

"""
Physics-Informed FNO (PINO)训练脚本说明：
- 这是使用有限差分方法计算物理损失的Physics-Informed FNO实现
- 使用标准FNO架构，通过PhysicsInformer计算PDE残差
- 结合数据损失和物理损失进行训练
- 适用于任何网络架构，不依赖自动微分
"""


def validation_step(model, dataloader, epoch):
    """
    验证步骤函数
    
    在验证集上评估模型性能，生成可视化结果。
    
    Parameters
    ----------
    model : FNO
        训练好的FNO模型
    dataloader : DataLoader
        验证数据加载器
    epoch : int
        当前epoch数
        
    Returns
    -------
    float
        平均验证损失
    """
    model.eval()  # 设置为评估模式

    with torch.no_grad():  # 禁用梯度计算
        loss_epoch = 0
        # 遍历验证数据
        for data in dataloader:
            invar, outvar, _, _ = data  # 解包数据：输入、输出、x坐标、y坐标
            # 模型前向传播：只使用第一个通道的渗透率
            out = model(invar[:, 0].unsqueeze(dim=1))
            # 累积MSE损失
            loss_epoch += F.mse_loss(outvar, out)

        # ===================================================================
        # 数据可视化
        # ===================================================================
        # 转换为numpy数组用于可视化
        outvar = outvar.detach().cpu().numpy()
        predvar = out.detach().cpu().numpy()

        # 创建1行3列的子图
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        # 计算颜色范围，确保真实值和预测值使用相同的颜色映射
        d_min = np.min(outvar[0, 0])
        d_max = np.max(outvar[0, 0])

        # 绘制真实值
        im = ax[0].imshow(outvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[0])
        
        # 绘制预测值
        im = ax[1].imshow(predvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[1])
        
        # 绘制误差图
        im = ax[2].imshow(np.abs(predvar[0, 0] - outvar[0, 0]))
        plt.colorbar(im, ax=ax[2])

        # 设置子图标题
        ax[0].set_title("True")
        ax[1].set_title("Pred")
        ax[2].set_title("Difference")

        # 保存图像
        fig.savefig(f"results_{epoch}.png")
        plt.close()
        
        # 返回平均损失
        return loss_epoch / len(dataloader)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):
    """
    Physics-Informed FNO (PINO)主训练函数
    
    使用有限差分方法计算物理损失的Physics-Informed FNO训练。
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra配置对象，包含所有训练参数
    """
    # ===================================================================
    # 1. 设备设置
    # ===================================================================
    # 检查CUDA可用性并设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 初始化日志系统
    LaunchLogger.initialize()

    # ===================================================================
    # 2. 物理方程定义
    # ===================================================================
    # 使用Diffusion方程表示Darcy方程
    # Darcy方程：∇·(k∇u) = f
    # 其中：u是压力场，k是渗透率场，f是源项
    forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03  # 源项（经过缩放）
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
    
    """
    物理方程说明：
    - T="u": 温度/压力场变量名为"u"
    - time=False: 稳态问题，不包含时间项
    - dim=2: 二维问题
    - D="k": 扩散系数为渗透率场"k"
    - Q=forcing_fn: 源项
    """

    # ===================================================================
    # 3. 数据加载
    # ===================================================================
    # 加载训练数据集
    dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/train.hdf5"), device=device
    )
    
    # 加载验证数据集
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device=device
    )

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    # ===================================================================
    # 4. 模型定义
    # ===================================================================
    # 创建FNO模型
    model = FNO(
        in_channels=cfg.model.fno.in_channels,      # 输入通道数：1（渗透率）
        out_channels=cfg.model.fno.out_channels,    # 输出通道数：1（压力）
        decoder_layers=cfg.model.fno.decoder_layers, # 解码器层数：1
        decoder_layer_size=cfg.model.fno.decoder_layer_size, # 解码器层大小：32
        dimension=cfg.model.fno.dimension,           # 问题维度：2D
        latent_channels=cfg.model.fno.latent_channels, # 潜在通道数：32
        num_fno_layers=cfg.model.fno.num_fno_layers, # FNO层数：4
        num_fno_modes=cfg.model.fno.num_fno_modes,   # Fourier模式数：12
        padding=cfg.model.fno.padding,               # 填充大小：9
    ).to(device)  # 移动到指定设备

    # ===================================================================
    # 5. 物理信息器设置
    # ===================================================================
    # 创建PhysicsInformer，用于计算PDE残差
    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],           # 需要的输出：扩散方程残差
        equations=darcy,                           # 物理方程
        grad_method="finite_difference",           # 梯度计算方法：有限差分
        device=device,                             # 计算设备
        fd_dx=1 / 240,                            # 有限差分网格间距（单位正方形，分辨率240）
    )
    
    """
    PhysicsInformer说明：
    - required_outputs: 指定需要计算的PDE残差类型
    - equations: 物理方程定义
    - grad_method: 梯度计算方法（有限差分 vs 自动微分）
    - fd_dx: 有限差分的网格间距，用于计算偏导数
    """

    # ===================================================================
    # 6. 优化器和学习率调度器
    # ===================================================================
    # 创建Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),        # Adam参数
        lr=cfg.start_lr,           # 初始学习率
        weight_decay=0.0,          # 权重衰减
    )

    # 创建指数衰减学习率调度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # ===================================================================
    # 7. 训练循环
    # ===================================================================
    for epoch in range(cfg.max_epochs):
        # 使用LaunchLogger包装epoch，用于控制台日志记录
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            # 遍历训练数据
            for data in dataloader:
                optimizer.zero_grad()  # 清零梯度
                
                # 解包数据
                invar = data[0]    # 输入变量（渗透率场）
                outvar = data[1]   # 输出变量（压力场）

                # ===================================================================
                # 前向传播
                # ===================================================================
                # 模型前向传播：只使用第一个通道的渗透率
                out = model(invar[:, 0].unsqueeze(dim=1))

                # ===================================================================
                # 物理损失计算
                # ===================================================================
                # 使用PhysicsInformer计算PDE残差
                residuals = phy_informer.forward(
                    {
                        "u": out,                    # 模型预测的压力场
                        "k": invar[:, 0:1],         # 输入渗透率场
                    }
                )
                pde_out_arr = residuals["diffusion_u"]  # 获取扩散方程残差

                # 边界条件处理：在边界处填充零值
                pde_out_arr = F.pad(
                    pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                
                # 计算物理损失：残差应该为零
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # ===================================================================
                # 数据损失计算
                # ===================================================================
                # 计算数据拟合损失
                loss_data = F.mse_loss(outvar, out)

                # ===================================================================
                # 总损失计算
                # ===================================================================
                # 组合数据损失和物理损失
                # 物理损失权重包含网格间距因子 1/240
                loss = loss_data + 1 / 240 * cfg.physics_weight * loss_pde

                # ===================================================================
                # 反向传播和优化
                # ===================================================================
                loss.backward()        # 反向传播
                optimizer.step()       # 更新参数
                scheduler.step()       # 更新学习率
                
                # 记录损失
                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                )

            # 记录epoch级别的学习率
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # ===================================================================
        # 8. 验证和检查点保存
        # ===================================================================
        # 验证阶段
        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataloader, epoch)
            log.log_epoch({"Validation error": error})

        # 保存检查点
        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
