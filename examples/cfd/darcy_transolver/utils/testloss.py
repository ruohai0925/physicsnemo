# ignore_header_test
# ruff: noqa: E402
""""""

"""
Transolver model. This code was modified from, https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch


class TestLoss(object):
    """
    测试损失函数类
    
    用于计算PDE求解的相对误差损失，特别适用于Transolver等PDE求解器。
    支持绝对L2误差和相对L2误差的计算。
    
    参数:
        d: 空间维度，默认为2（2D问题）
        p: 范数类型，默认为2（L2范数）
        size_average: 是否对批次大小求平均
        reduction: 是否进行归约操作
    """
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        """
        初始化测试损失函数
        
        参数:
            d: 空间维度（必须大于0）
            p: 范数类型（必须大于0）
            size_average: 是否对批次大小求平均
            reduction: 是否进行归约操作
        """
        super(TestLoss, self).__init__()

        assert d > 0 and p > 0  # 确保维度和范数参数有效

        self.d = d                    # 空间维度
        self.p = p                    # 范数类型
        self.reduction = reduction    # 归约标志
        self.size_average = size_average  # 平均标志

    def abs(self, x, y):
        """
        计算绝对L2误差
        
        参数:
            x: 预测值张量
            y: 真实值张量
            
        返回:
            绝对L2误差，考虑网格间距的权重
        """
        num_examples = x.size()[0]  # 批次大小

        # 计算网格间距（假设均匀网格）
        h = 1.0 / (x.size()[1] - 1.0)

        # 计算加权L2范数误差
        # 权重 h^(d/p) 考虑了空间维度和范数类型
        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        # 根据配置进行归约
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)  # 对批次求平均
            else:
                return torch.sum(all_norms)   # 对批次求和

        return all_norms  # 返回每个样本的误差

    def rel(self, x, y):
        """
        计算相对L2误差
        
        参数:
            x: 预测值张量
            y: 真实值张量
            
        返回:
            相对L2误差：||x-y||_p / ||y||_p
        """
        num_examples = x.size()[0]  # 批次大小

        # 计算预测值和真实值之间的L2范数
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        # 计算真实值的L2范数
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        
        # 根据配置进行归约
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)  # 对批次求平均
            else:
                return torch.sum(diff_norms / y_norms)   # 对批次求和

        return diff_norms / y_norms  # 返回每个样本的相对误差

    def __call__(self, x, y):
        """
        调用函数，默认使用相对误差
        
        参数:
            x: 预测值张量
            y: 真实值张量
            
        返回:
            相对L2误差
        """
        return self.rel(x, y)
