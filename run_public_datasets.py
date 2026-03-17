"""
公开数据集实验入口 - 调用 standalone 文件
==============================================
不再自行训练, 改为调用各自独立的 standalone 脚本。

Usage:
    python run_public_datasets.py --dataset taiwan
    python run_public_datasets.py --dataset nhtsa
    python run_public_datasets.py --dataset all
"""

import os
import sys
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_taiwan(data_file='Restaurant Complaint balanced.xlsx', mode='all'):
    """调用 run_taiwan_restaurant_standalone"""
    print("\n" + "=" * 60)
    print("🍜 Taiwan Restaurant Dataset (standalone)")
    print("=" * 60)

    taiwan_mod = importlib.import_module('run_taiwan_restaurant_standalone')

    if mode in ['baseline', 'all']:
        exp = taiwan_mod.TaiwanBaselineExperiment(data_file)
        exp.run_baseline()

    if mode in ['ablation', 'all']:
        taiwan_mod.run_taiwan_ablation(data_file)


def run_nhtsa(data_file='NHTSA_processed.xlsx', mode='all'):
    """调用 run_nhtsa_standalone"""
    print("\n" + "=" * 60)
    print("🚗 NHTSA Vehicle Complaint Dataset (standalone)")
    print("=" * 60)

    nhtsa_mod = importlib.import_module('run_nhtsa_standalone')

    if mode in ['baseline', 'all']:
        exp = nhtsa_mod.NHTSABaselineExperiment(data_file)
        exp.run_baseline()

    if mode in ['ablation', 'all']:
        nhtsa_mod.run_nhtsa_ablation(data_file)


def main():
    parser = argparse.ArgumentParser(description='公开数据集实验 (调用standalone)')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['taiwan', 'nhtsa', 'all'],
                        help='数据集选择')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'baseline', 'ablation'],
                        help='运行模式')
    parser.add_argument('--taiwan_file', type=str,
                        default='Restaurant Complaint balanced.xlsx',
                        help='台湾餐厅数据文件')
    parser.add_argument('--nhtsa_file', type=str,
                        default='NHTSA_processed.xlsx',
                        help='NHTSA数据文件')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("公开数据集实验 (standalone 模式)")
    print("=" * 60)
    print(f"工作目录: {os.getcwd()}")

    results = {}

    if args.dataset in ['taiwan', 'all']:
        if os.path.exists(args.taiwan_file):
            run_taiwan(args.taiwan_file, args.mode)
        else:
            print(f"⚠️ 文件不存在: {args.taiwan_file}")

    if args.dataset in ['nhtsa', 'all']:
        if os.path.exists(args.nhtsa_file):
            run_nhtsa(args.nhtsa_file, args.mode)
        else:
            print(f"⚠️ 文件不存在: {args.nhtsa_file}")

    print("\n" + "=" * 60)
    print("✅ 公开数据集实验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
