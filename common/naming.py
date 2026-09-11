"""命名辅助：时间戳名称 + 名称冲突去重（追加 -1、-2…，遵循项目规范）。"""
from datetime import datetime
from pathlib import Path


def timestamp(fmt: str = "%Y%m%d%H%M") -> str:
    """当前时间字符串，如 202609091135（年月日时分）。"""
    return datetime.now().strftime(fmt)


def unique_path(path) -> Path:
    """若 path 已存在，则在末尾追加 -1、-2 … 后返回（文件/文件夹均适用）。

    例：model_202609091135 已存在 -> model_202609091135-1
        result.txt 已存在       -> result-1.txt
    """
    path = Path(path)
    if not path.exists():
        return path
    k = 1
    while True:
        cand = path.with_name(f"{path.stem}-{k}{path.suffix}")
        if not cand.exists():
            return cand
        k += 1


def model_folder_name(ts: str) -> str:
    """模型文件夹名，如 model_202609091135。"""
    return f"model_{ts}"
