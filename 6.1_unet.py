"""
这个文件用于执行 project.md 的 6.1：
Train/calibrate an organ segmentation model on CT-ORG。

作用：
- 提供阶段化入口（6.1_unet.py），方便直接运行。
- 统一调用 6.1_seg_model.py 里的完整 6.1 训练与校准流程，避免重复实现。

输入：
- 命令行参数（与 6.1_seg_model.py 保持一致）
- data/PKG - CT-ORG/CT-ORG/OrganSegmentations/*

输出：
- output/experiments/organ_seg/<run_tag>/...
"""
import runpy
from pathlib import Path


def main():
    """Why: 在 6.1_unet.py 中启动 CT-ORG 器官分割训练流程。

    Content: 调用 6.1_seg_model.py 主流程，执行 6.1 训练与校准。
    Input: 命令行参数（透传给 6.1_seg_model.py）。
    Output: 生成模型、训练日志与 token 文件。
    """
    seg_script = Path(__file__).with_name("6.1_seg_model.py")
    if not seg_script.exists():
        raise SystemExit(f"missing script: {seg_script}")
    runpy.run_path(str(seg_script), run_name="__main__")


if __name__ == "__main__":
    main()
