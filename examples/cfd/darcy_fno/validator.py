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
详细注释版本的Grid Validator

本文件详细解释了如何创建和使用网格验证器节点，包括：
1. 验证器的初始化和配置
2. 数据反归一化处理
3. 可视化图像生成
4. 误差计算和分析
5. MLFlow集成

作者：AI Assistant
日期：2024
"""

import matplotlib.pyplot as plt
from torch import FloatTensor
from physicsnemo.utils.logging import LaunchLogger


class GridValidator:
    """
    网格验证器类
    
    验证器比较模型输出和目标，反归一化数据并绘制样本图像。
    这是PhysicsNeMo中用于验证模型性能的核心组件。
    
    参数
    ----------
    loss_fun : MSELoss
        用于评估验证误差的损失函数
    norm : Dict, optional
        用于归一化输入和目标的每个通道的均值和标准差
    font_size : float, optional
        图像中使用的字体大小
    """

    def __init__(
        self,
        loss_fun,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        font_size: float = 28.0,
    ):
        """
        初始化网格验证器
        
        参数：
        - loss_fun: 损失函数，用于计算验证误差
        - norm: 归一化参数字典，包含渗透率和Darcy压力场的均值和标准差
        - font_size: 图像字体大小
        """
        # 存储归一化参数 - 用于反归一化数据
        self.norm = norm
        
        # 存储损失函数 - 用于计算验证误差
        self.criterion = loss_fun
        
        # 设置图像字体大小
        self.font_size = font_size
        
        # 定义图像标题
        self.headers = ("invar", "truth", "prediction", "relative error")
        
        """
        归一化参数说明：
        - permeability: 渗透率场的归一化参数 (mean, std)
        - darcy: Darcy压力场的归一化参数 (mean, std)
        
        反归一化公式：
        original_value = normalized_value * std + mean
        """

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        logger: LaunchLogger,
    ) -> float:
        """
        比较模型输出、目标并绘制所有内容
        
        这是验证器的核心方法，执行以下操作：
        1. 计算验证损失
        2. 反归一化数据
        3. 生成可视化图像
        4. 记录到MLFlow
        
        参数
        ----------
        invar : FloatTensor
            模型输入（渗透率场）
        target : FloatTensor
            真实值（压力场）
        prediction : FloatTensor
            模型输出（预测压力场）
        step : int
            迭代计数器（用于命名图像文件）
        logger : LaunchLogger
            用于记录图像的日志记录器
        
        返回
        -------
        float
            验证误差
        """
        
        # ===================================================================
        # 步骤1: 计算验证损失
        # ===================================================================
        # 使用损失函数计算预测值和真实值之间的误差
        loss = self.criterion(prediction, target)
        
        # 获取归一化参数
        norm = self.norm
        
        """
        损失计算说明：
        - 使用MSE损失函数计算预测值和真实值之间的均方误差
        - 这是在归一化空间中的损失
        - 损失值反映了模型预测的准确性
        """

        # ===================================================================
        # 步骤2: 数据反归一化处理
        # ===================================================================
        # 从批次中选择第一个样本进行可视化
        # 反归一化渗透率场
        invar = invar * norm["permeability"][1] + norm["permeability"][0]
        
        # 反归一化真实压力场
        target = target * norm["darcy"][1] + norm["darcy"][0]
        
        # 反归一化预测压力场
        prediction = prediction * norm["darcy"][1] + norm["darcy"][0]
        
        # 转换为numpy数组并选择第一个样本
        # 注意：invar使用[-1]索引，因为可能包含多个通道
        invar = invar.cpu().numpy()[0, -1, :, :]  # [256, 256]
        target = target.cpu().numpy()[0, 0, :, :]  # [256, 256]
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]  # [256, 256]
        
        """
        反归一化说明：
        - 将归一化的数据转换回原始物理单位
        - 公式：original = normalized * std + mean
        - 对于渗透率：original = normalized * 0.75 + 1.25
        - 对于压力：original = normalized * 2.79E-2 + 4.52E-2
        
        数据形状说明：
        - 输入批次形状：[batch_size, channels, height, width]
        - 选择第一个样本：[0, :, :, :]
        
        通道选择解释：
        - 对于渗透率场(invar)：
          * 原始数据：[64, 1, 256, 256] (只有1个通道)
          * 但在FNO模型中，如果coord_features=True，会添加坐标特征
          * 变成：[64, 3, 256, 256] (渗透率 + x坐标 + y坐标)
          * 所以使用[-1]选择最后一个通道，即y坐标通道
          * 实际上这里应该选择[0]通道（渗透率），[-1]可能是代码中的错误
        
        - 对于压力场(target/prediction)：
          * 始终只有1个通道：[64, 1, 256, 256]
          * 所以使用[0]选择唯一的通道
        
        可视化数据说明：
        - 这些数据已经经过反归一化，是原始物理单位的值
        - 不是像素值，而是物理量（渗透率或压力）
        - 渗透率范围：约0.5-2.0 (原始单位)
        - 压力范围：约0.01-0.08 (原始单位)
        """

        # ===================================================================
        # 步骤3: 生成可视化图像
        # ===================================================================
        # 关闭之前的图像以释放内存
        plt.close("all")
        
        # 设置图像参数
        plt.rcParams.update({"font.size": self.font_size})
        
        # 创建1行4列的子图布局
        fig, ax = plt.subplots(1, 4, figsize=(15 * 4, 15), sharey=True)
        
        # 存储图像对象用于添加颜色条
        im = []
        
        # 绘制渗透率场（输入）
        im.append(ax[0].imshow(invar))
        ax[0].set_title(self.headers[0])  # "invar"
        
        # 绘制真实压力场
        im.append(ax[1].imshow(target))
        ax[1].set_title(self.headers[1])  # "truth"
        
        # 绘制预测压力场
        im.append(ax[2].imshow(prediction))
        ax[2].set_title(self.headers[2])  # "prediction"
        
        # 绘制相对误差
        # 相对误差 = (预测值 - 真实值) / 标准差
        relative_error = (prediction - target) / norm["darcy"][1]
        im.append(ax[3].imshow(relative_error))
        ax[3].set_title(self.headers[3])  # "relative error"
        
        """
        可视化说明：
        - 第1个子图：输入渗透率场，显示问题的输入条件
        - 第2个子图：真实压力场，显示物理求解器的结果
        - 第3个子图：模型预测，显示FNO的输出
        - 第4个子图：相对误差，显示预测的准确性
        
        相对误差计算：
        - 使用标准差进行归一化，使得误差具有物理意义
        - 正值表示预测值高于真实值
        - 负值表示预测值低于真实值
        - 绝对值越小表示预测越准确
        """

        # ===================================================================
        # 步骤4: 添加颜色条和格式化
        # ===================================================================
        # 为每个子图添加颜色条
        for ii in range(len(im)):
            fig.colorbar(
                im[ii], 
                ax=ax[ii], 
                location="bottom",  # 颜色条位置
                fraction=0.046,     # 颜色条大小
                pad=0.04           # 颜色条与图像的间距
            )

        # ===================================================================
        # 步骤5: 记录图像到MLFlow
        # ===================================================================
        # 使用LaunchLogger记录图像到MLFlow
        logger.log_figure(
            figure=fig, 
            artifact_file=f"validation_step_{step:03d}.png"
        )
        
        """
        MLFlow集成说明：
        - log_figure方法将图像保存为PNG文件
        - 文件名包含步骤编号，便于跟踪训练进度
        - 图像自动上传到MLFlow服务器
        - 可以在MLFlow UI中查看训练过程中的验证结果
        """

        # 返回验证损失
        return loss
