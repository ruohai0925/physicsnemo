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
from itertools import chain
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils import HDF5MapStyleDataset

"""
Physics-Informed DeepONet训练脚本说明：
- 这是使用自动微分方法计算物理损失的Physics-Informed DeepONet实现
- 使用FNO作为Branch网络，FC作为Trunk网络
- 通过PhysicsNeMo Sym的自动微分计算PDE残差
- 需要坐标输入设置requires_grad=True以支持梯度计算
"""


def validation_step(graph, dataloader, epoch):
    """
    DeepONet验证步骤函数
    
    在验证集上评估DeepONet模型性能，生成可视化结果。
    
    Parameters
    ----------
    graph : MdlsSymWrapper
        训练好的DeepONet模型（包装器）
    dataloader : DataLoader
        验证数据加载器
    epoch : int
        当前epoch数
        
    Returns
    -------
    float
        平均验证损失
    """

    with torch.no_grad():  # 禁用梯度计算
        loss_epoch = 0
        # 遍历验证数据
        for data in dataloader:
            invar, outvar, x_invar, y_invar = data  # 解包数据
            
            # DeepONet前向传播
            out = graph.forward(
                {
                    "k_prime": invar[:, 0].unsqueeze(dim=1),  # 渗透率输入
                    "x": x_invar,                             # x坐标
                    "y": y_invar                              # y坐标
                }
            )

            deepo_out_u = out["u"]  # 获取压力场输出

            # 累积MSE损失
            loss_epoch += F.mse_loss(outvar, deepo_out_u)

        # ===================================================================
        # 数据可视化
        # ===================================================================
        # 转换为numpy数组用于可视化
        outvar = outvar.detach().cpu().numpy()
        predvar = deepo_out_u.detach().cpu().numpy()

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


class MdlsSymWrapper(Arch):
    """
    Wrapper model to convert PhysicsNeMo model to PhysicsNeMo-Sym model.

    PhysicsNeMo Sym relies on the inputs/outputs of the model being dictionary of tensors.
    This wrapper converts the input dictionary of tensors to a tensor inputs that can
    be processed by the PhysicsNeMo model that operate on tensors. Appropriate
    transformations are performed in the forward pass of the model to translate between
    these two input/output definitions.

    These transformations can differ based on the models. For e.g. typically for a fully
    connected network, the input tensors are combined by concatenating them along
    appropriate dimension before passing them as an input to the PhysicsNeMo model.
    During the output, the process is reversed, the output tensor from pytorch model is
    split across appropriate dimensions and then converted to a dictionary with
    appropriate keys to produce the final output.

    Having the model wrapped in a wrapper like this allows gradient computation using
    the PhysicsNeMo Sym's optimized gradient computing backend.

    For more details on PhysicsNeMo Sym models, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/tutorials/simple_training_example.html#using-custom-models-in-physicsnemo
    For more details on Key class, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.html#module-physicsnemo.sym.key
    """

    def __init__(
        self,
        input_keys=[Key("k"), Key("x"), Key("y")],
        output_keys=[Key("k_prime"), Key("u")],
        trunk_net=None,
        branch_net=None,
    ):
        """
        初始化DeepONet包装器
        
        Parameters
        ----------
        input_keys : list
            输入键列表，定义模型输入
        output_keys : list
            输出键列表，定义模型输出
        trunk_net : FullyConnected
            Trunk网络，处理坐标信息
        branch_net : FNO
            Branch网络，处理渗透率场
        """
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
        )

        self.branch_net = branch_net  # FNO网络（Branch）
        self.trunk_net = trunk_net    # 全连接网络（Trunk）

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        """
        DeepONet前向传播
        
        实现DeepONet的前向传播：
        1. Trunk网络处理坐标信息
        2. Branch网络处理渗透率场
        3. 两个网络的输出相乘得到最终结果
        
        Parameters
        ----------
        dict_tensor : Dict[str, torch.Tensor]
            输入张量字典，包含"k_prime"（渗透率）、"x"（x坐标）、"y"（y坐标）
            
        Returns
        -------
        Dict[str, torch.Tensor]
            输出张量字典，包含"k"（渗透率）和"u"（压力）
        """
        # ===================================================================
        # 1. Trunk网络处理坐标信息
        # ===================================================================
        # 获取坐标输入的形状
        xy_input_shape = dict_tensor["x"].shape
        
        # 将x, y坐标连接起来，输入到Trunk网络（MLP）
        xy = self.concat_input(
            {
                k: dict_tensor[k].view(xy_input_shape[0], -1, 1) for k in ["x", "y"]
            },  # 展平坐标维度
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,  # 沿最后一个维度连接以形成特征向量
        )
        
        # Trunk网络前向传播
        fc_out = self.trunk_net(xy)

        # ===================================================================
        # 2. Branch网络处理渗透率场
        # ===================================================================
        # 将渗透率场输入到Branch网络（FNO）
        fno_out = self.branch_net(dict_tensor["k_prime"])

        # ===================================================================
        # 3. 输出组合
        # ===================================================================
        # 重新整形fc_out以匹配空间维度
        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )

        # 将Branch和Trunk网络的输出相乘得到最终输出
        # 这是DeepONet的核心操作：u(x,y) = Σ(branch_i * trunk_i(x,y))
        out = fc_out * fno_out

        # 沿通道维度分割输出，得到张量字典
        return self.split_output(
            out, self.output_key_dict, dim=1
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config_deeponet.yaml")
def main(cfg: DictConfig):
    """
    Physics-Informed DeepONet主训练函数
    
    使用自动微分方法计算物理损失的Physics-Informed DeepONet训练。
    
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

    dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/train.hdf5"), device=device
    )
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device=device
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model_branch = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    )

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    )

    # Define k-prime as an auxiliary variable that is a copy of k.
    # Having k as the output of the model will allow gradients of k (for pde loss)
    # to be computed using Sym's gradient backend
    model = MdlsSymWrapper(
        input_keys=[Key("k_prime"), Key("x"), Key("y")],
        output_keys=[Key("k"), Key("u")],
        trunk_net=model_trunk,
        branch_net=model_branch,
    ).to(device)

    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy,
        grad_method="autodiff",
        device=device,
    )

    optimizer = torch.optim.Adam(
        chain(model_branch.parameters(), model_trunk.parameters()),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
        fused=True if torch.cuda.is_available() else False,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                outvar = data[1]

                coords = torch.stack([data[2], data[3]], dim=1).requires_grad_(True)
                # compute forward pass
                out = model.forward(
                    {
                        "k_prime": data[0][:, 0].unsqueeze(dim=1),
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    }
                )

                residuals = phy_informer.forward(
                    {
                        "coordinates": coords,
                        "u": out["u"],
                        "k": out["k"],
                    }
                )
                pde_out_arr = residuals["diffusion_u"]

                # Boundary condition
                pde_out_arr = F.pad(
                    pde_out_arr[..., 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # Compute data loss
                deepo_out_u = out["u"]
                deepo_out_k = out["k"]
                loss_data = F.mse_loss(outvar, deepo_out_u) + F.mse_loss(
                    data[0][:, 0].unsqueeze(dim=1), deepo_out_k
                )

                # Compute total loss
                loss = loss_data + cfg.physics_weight * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataloader, epoch)
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./checkpoints",
            models=[model_branch, model_trunk],
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
