"""
统一日志配置模块
支持DDP训练环境，只在rank 0进程输出日志
"""
import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "d25_t6",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: Optional[int] = None
) -> logging.Logger:
    """
    设置并返回一个配置好的logger
    
    Args:
        name: logger名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        rank: 当前进程的rank（在DDP环境中使用）。如果为None，会自动检测。
               只有rank 0或非DDP环境才会输出日志到控制台
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 自动检测rank（如果未提供）
    if rank is None:
        rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    
    # 格式化字符串
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler（只在rank 0或非DDP环境输出）
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件handler（如果指定了日志文件，所有进程都写入）
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "d25_t6") -> logging.Logger:
    """
    获取已配置的logger，如果不存在则创建
    
    Args:
        name: logger名称
    
    Returns:
        logger实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果logger还没有配置，使用默认配置
        setup_logger(name)
    return logger


# 创建默认的logger实例
default_logger = setup_logger()

