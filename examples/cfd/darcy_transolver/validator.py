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
Darcy流问题的网格验证器

本模块实现了GridValidator类，用于验证Transolver模型的预测结果。
主要功能包括：
1. 计算预测误差和损失
2. 反归一化数据用于可视化
3. 生成对比图像（输入、真实值、预测值、相对误差）
4. 将结果保存到MLFlow日志系统

验证过程：
- 输入：渗透率场、真实压力场、预测压力场
- 输出：验证损失和可视化图像
- 可视化：4个子图显示输入、真实值、预测值和相对误差

作者：PhysicsNeMo团队
"""

import matplotlib.pyplot as plt
from torch import FloatTensor
from physicsnemo.utils.logging import LaunchLogger


class GridValidator:
    """
    网格验证器
    
    用于验证Transolver模型的预测结果，比较模型输出和目标值，
    进行反归一化处理并生成可视化图像。
    
    参数:
        loss_fun: 损失函数，用于评估验证误差
        norm: 字典，包含每个通道的均值和标准差，用于归一化输入和目标
        font_size: 图像中使用的字体大小
    """

    def __init__(
        self,
        loss_fun,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        font_size: float = 28.0,
    ):
        """
        初始化网格验证器
        
        参数:
            loss_fun: 损失函数实例
            norm: 归一化参数字典，格式为 {"permeability": (mean, std), "darcy": (mean, std)}
            font_size: 图像字体大小
        """
        self.norm = norm                    # 归一化参数
        self.criterion = loss_fun           # 损失函数
        self.font_size = font_size          # 字体大小
        self.headers = ("invar", "truth", "prediction", "relative error")  # 图像标题

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        logger: LaunchLogger,
    ) -> float:
        """
        比较模型输出和目标值，并生成可视化图像
        
        参数:
            invar: 模型输入（渗透率场）
            target: 真实目标值（压力场）
            prediction: 模型预测输出（压力场）
            step: 迭代计数器
            logger: 日志记录器，用于保存图像
            
        返回:
            float: 验证误差（损失值）
        """
        # 计算验证损失
        loss = self.criterion(prediction, target)
        norm = self.norm

        # 反归一化处理：将归一化的数据转换回原始尺度
        # 渗透率场反归一化
        invar = invar * norm["permeability"][1] + norm["permeability"][0]
        # 压力场反归一化
        target = target * norm["darcy"][1] + norm["darcy"][0]
        prediction = prediction * norm["darcy"][1] + norm["darcy"][0]
        
        # 选择批次中的第一个样本进行可视化
        invar = invar.cpu().numpy()[0, -1, :, :]      # 取最后一个通道的渗透率
        target = target.cpu().numpy()[0, 0, :, :]     # 取第一个通道的压力场
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]  # 取第一个通道的预测

        # 创建可视化图像
        plt.close("all")  # 关闭之前的图像
        plt.rcParams.update({"font.size": self.font_size})  # 设置字体大小
        
        # 创建1行4列的子图布局
        fig, ax = plt.subplots(1, 4, figsize=(15 * 4, 15), sharey=True)
        im = []
        
        # 绘制四个子图
        im.append(ax[0].imshow(invar))                    # 输入渗透率场
        im.append(ax[1].imshow(target))                   # 真实压力场
        im.append(ax[2].imshow(prediction))               # 预测压力场
        im.append(ax[3].imshow((prediction - target) / norm["darcy"][1]))  # 相对误差

        # 为每个子图添加颜色条和标题
        for ii in range(len(im)):
            fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
            ax[ii].set_title(self.headers[ii])

        # 将图像保存到MLFlow日志系统
        logger.log_figure(figure=fig, artifact_file=f"validation_step_{step:03d}.png")

        return loss
