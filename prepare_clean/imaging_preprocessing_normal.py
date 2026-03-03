"""
作用：
- 这是 imaging_preprocessing.py 的兼容入口脚本。
- 用于在 PyCharm pydevconsole 环境运行时，忽略 IDE 自动注入的未知参数。

输入：
- 可选参数 --max-cases N（默认 0，表示全量）
- 其他未知参数会被自动忽略（例如 --mode/--host/--port）

输出：
- 调用 imaging_preprocessing.run_pipeline 生成同样的预处理输出文件。
"""
import argparse

from imaging_preprocessing import run_pipeline


def parse_args():
    """Why: PyCharm 控制台会注入额外参数，直接 parse_args 会报错。

    Content: 只解析本脚本需要的参数，忽略未知参数。
    Input: 命令行参数。
    Output: argparse 参数对象（包含 max_cases）。
    """
    parser = argparse.ArgumentParser(description="Run imaging preprocessing (pydevconsole-safe).")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 means process all patients; >0 means process first N patients.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main():
    """Why: 提供一个稳定入口，避免 IDE 参数导致失败。

    Content: 解析参数并执行预处理 pipeline。
    Input: 命令行参数。
    Output: 生成预处理结果文件并打印统计信息。
    """
    args = parse_args()
    run_pipeline(args.max_cases)


if __name__ == "__main__":
    main()
