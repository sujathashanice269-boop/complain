"""
一键启动脚本 - run.py
让代码新手可以轻松运行完整训练流程

新增选项:
  选项1: 一键全流程 (从预训练到最终可视化，全自动)
  选项2-5: 快速测试 (验证代码)
  选项6-9: 正式训练 (分步执行)
"""

import os
import sys
import time

# ============================================================
# 环境变量设置（确保子进程继承）
# ============================================================
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def get_python_cmd():
    """获取当前 Python 解释器路径"""
    return sys.executable


def run_cmd(cmd, desc=None):
    """执行命令并打印"""
    if desc:
        print(f"\n{'='*70}")
        print(f"  {desc}")
        print(f"{'='*70}")
    print(f"  命令: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  ⚠️ 命令返回码: {ret}")
    return ret


def print_banner():
    print("\n" + "="*70)
    print("        客户重复投诉预测系统 - 三模态融合完全改进版")
    print("="*70)
    print("改进内容:")
    print("  ✓ 方向一: Text预训练 (30轮MLM + 20轮对比学习)")
    print("  ✓ 方向二: Label全局图预训练")
    print("  ✓ 方向三: 结构化特征重要性加权")
    print("  ✓ 方向四: 真正的跨模态注意力")
    print("  ✓ 方向五: 课程学习训练策略")
    print("  ✓ 方向六: 模态平衡损失")
    print("  数据集: 移动客户(私有) + 台湾餐厅 + NHTSA车辆投诉")
    print("="*70 + "\n")


def check_files():
    """检查必要文件"""
    print("🔍 检查必要文件...")

    required_files = {
        '数据文件': '小案例ai问询.xlsx',
        '大数据文件': '多模态初始表_数据标签.xlsx',
        '用户词典': 'new_user_dict.txt',
        '模型文件': 'model.py',
        '配置文件': 'config.py',
        '数据处理': 'data_processor.py',
        '主程序': 'main.py'
    }

    optional_files = {
        '台湾餐厅数据': 'Restaurant Complaint balanced.xlsx',
        'NHTSA数据': 'NHTSA_processed.xlsx',
        '台湾standalone': 'run_taiwan_restaurant_standalone.py',
        'NHTSA standalone': 'run_nhtsa_standalone.py',
    }

    missing = []
    for name, file in required_files.items():
        if os.path.exists(file):
            print(f"  ✓ {name}: {file}")
        else:
            print(f"  ✗ {name}: {file} (缺失)")
            missing.append(file)

    print()
    for name, file in optional_files.items():
        if os.path.exists(file):
            print(f"  ✓ {name}: {file}")
        else:
            print(f"  ○ {name}: {file} (可选，缺失)")

    if missing:
        print(f"\n❌ 缺少核心文件: {', '.join(missing)}")
        print("请确保所有必要文件都在当前目录下")
        return False

    print("\n✅ 核心文件齐全!\n")
    return True


def show_menu():
    """显示菜单"""
    print("\n请选择运行模式:")
    print("="*70)
    print("【一键全流程】")
    print("  1. ⭐ 一键全流程 (预训练→训练→基线→消融→可视化，全自动)")
    print()
    print("【快速测试模式】")
    print("  2. 🚀 完整流程快速测试 (1轮预训练+1轮训练，约14小时)")
    print("  3. 🔍 单独测试Text模型 (约47分钟)")
    print("  4. 🔍 单独测试Label模型 (约3分钟) ← 推荐先测这个!")
    print("  5. 🔍 单独测试Struct模型 (约3分钟)")
    print()
    print("【正式训练模式 (分步执行)】")
    print("  6. 完整预训练 (Text 30+20轮 + Label 20轮)")
    print("  7. 完整训练 (预训练 + 课程学习训练)")
    print("  8. 跳过预训练直接训练")
    print("  9. 生产环境完整流程 (最佳效果)")
    print()
    print("  0. 退出")
    print("="*70)

    choice = input("\n请输入选项 (0-9): ").strip()
    return choice


# ============================================================
# 选项1: 一键全流程
# ============================================================
def run_full_pipeline():
    """
    一键全流程: 从预训练到最终可视化，全自动执行
    无需手动设置任何脚本形参
    """
    python_cmd = get_python_cmd()

    print("\n" + "="*70)
    print("⭐ 一键全流程 - 自动执行以下步骤:")
    print("="*70)
    print("  ① main.py          → 移动客户预训练 + 课程学习训练")
    print("  ② baseline_all_methods.py → 移动客户5层基线 + TM-CRPP")
    print("  ③ run_public_datasets.py  → 台湾+NHTSA 基线+消融")
    print("  ④ ablation_study.py       → 移动客户消融实验")
    print("  ⑤ fusion_models.py            → 融合机制消融实验")
    print("  ⑥ cross_dataset_experiments.py → 三数据集跨域可视化")
    print("  ⑦ run_supplementary_experiments.py → 补充实验")
    print("  ⑧ interpretability_analysis.py → 可解释性分析")
    print("  ⑨ visualize_comprehensive_v2.py → 三模态对齐图")
    print("  ⑩ results_visualization.py → 最终结果汇总图")
    print("="*70)
    print()

    confirm = input("确认运行全流程? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    start_time = time.time()
    step = 0
    total_steps = 10

    def step_header(desc):
        nonlocal step
        step += 1
        print(f"\n{'#'*70}")
        print(f"# 步骤 {step}/{total_steps}: {desc}")
        print(f"{'#'*70}\n")

    # ① 移动客户预训练 + 课程学习训练
    step_header("移动客户预训练 + 课程学习训练 (main.py)")
    run_cmd(f'"{python_cmd}" main.py --mode train --quick_test')

    # ② 移动客户5层基线实验
    step_header("移动客户5层基线实验 (baseline_all_methods.py)")
    run_cmd(f'"{python_cmd}" baseline_all_methods.py --dataset_name default')

    # ③ 公开数据集 (台湾 + NHTSA) 基线+消融
    step_header("公开数据集基线+消融 (run_public_datasets.py)")
    run_cmd(f'"{python_cmd}" run_public_datasets.py --dataset all')

    # ④ 移动客户消融实验
    step_header("移动客户消融实验 (ablation_study.py)")
    run_cmd(f'"{python_cmd}" ablation_study.py --dataset telecom')

    step_header("融合机制消融实验 (fusion_models.py)")
    run_cmd(f'"{python_cmd}" fusion_models.py --fusion_type all --epochs 10')
    # ⑤ 三数据集跨域可视化
    step_header("三数据集跨域可视化 (cross_dataset_experiments.py)")
    run_cmd(f'"{python_cmd}" cross_dataset_experiments.py --experiment all')

    # ⑥ 补充实验
    step_header("补充实验 (run_supplementary_experiments.py)")
    run_cmd(f'"{python_cmd}" run_supplementary_experiments.py --exp unique_only')

    # ⑦ 可解释性分析
    step_header("可解释性分析 (interpretability_analysis.py)")
    run_cmd(f'"{python_cmd}" interpretability_analysis.py '
            f'--model_path ./outputs/baseline_comparison/default/tmcrpp_models/tmcrpp_default.pth '
            f'--data_file 小案例ai问询.xlsx '
            f'--ablation_results ablation_results_default.json')

    # ⑧ 三模态对齐图
    step_header("三模态对齐图 (visualize_comprehensive_v2.py)")
    run_cmd(f'"{python_cmd}" visualize_comprehensive_v2.py '
            f'--model_path ./outputs/baseline_comparison/default/tmcrpp_models/tmcrpp_default.pth '
            f'--data_file 小案例ai问询.xlsx '
            f'--new_code "_70&&0c&a#9aa996c-20240306-1"')

    # ⑨ 最终结果汇总图
    step_header("最终结果汇总图 (results_visualization.py)")
    run_cmd(f'"{python_cmd}" results_visualization.py '
            f'--results_file all_results.json '
            f'--mode all '
            f'--output_dir ./outputs')

    # 完成汇总
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print("\n" + "="*70)
    print("✅ 全流程执行完毕!")
    print(f"⏱️  总耗时: {hours}小时{minutes}分钟")
    print("="*70)
    print("\n📂 输出目录:")
    print("  - ./outputs/baseline_comparison/default/  (移动客户基线)")
    print("  - ./outputs/baseline_comparison/taiwan/    (台湾餐厅)")
    print("  - ./outputs/baseline_comparison/nhtsa/     (NHTSA车辆)")
    print("  - ./outputs/cross_dataset/                 (跨数据集可视化)")
    print("  - ./supplementary_results/                 (补充实验)")
    print("  - ./outputs/reports/                       (可解释性报告)")
    print("  - ./models/                                (保存的模型)")
    print()


# ============================================================
# 选项2-5: 快速测试
# ============================================================
def run_full_quick_test():
    """完整流程快速测试"""
    print("\n🚀 运行完整流程快速测试...")
    print("="*70)
    print("测试内容:")
    print("  1️⃣ Text预训练阶段1 (MLM) - 1轮")
    print("  2️⃣ Text预训练阶段2 (对比学习) - 1轮")
    print("  3️⃣ Label全局图预训练 - 1轮")
    print("  4️⃣ 课程学习阶段1 (单模态) - 1轮")
    print("  5️⃣ 课程学习阶段2 (双模态) - 1轮")
    print("  6️⃣ 课程学习阶段3 (三模态) - 1轮")
    print()
    print("⏱️  预计时间: 约14小时")
    print("💡 目的: 验证整个训练流程能否跑通")
    print("="*70)

    confirm = input("\n确认运行? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode train --quick_test')


def run_test_text_only():
    """单独测试Text"""
    print("\n🔍 单独测试Text模型...")
    print("配置: 只训练text_only模型 (约47分钟)")
    print("用途: 验证BERT预训练和文本处理是否正常\n")
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode train --quick_test --test_single_modal text')


def run_test_label_only():
    """单独测试Label"""
    print("\n🔍 单独测试Label模型...")
    print("配置: 只训练label_only模型 (约3分钟)")
    print("用途: 快速验证GAT标签编码和全局图预训练是否正常")
    print("💡 推荐: 先测试这个，快速定位问题!\n")
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode train --quick_test --test_single_modal label')


def run_test_struct_only():
    """单独测试Struct"""
    print("\n🔍 单独测试Struct模型...")
    print("配置: 只训练struct_only模型 (约3分钟)")
    print("用途: 验证结构化特征处理是否正常\n")
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode train --quick_test --test_single_modal struct')


# ============================================================
# 选项6-9: 正式训练
# ============================================================
def run_full_pretrain():
    """完整预训练"""
    print("\n📚 运行完整预训练...")
    print("配置: Text(30+20轮) + Label(20轮) (约2-4小时)\n")
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode pretrain_only --production')


def run_full_train():
    """完整训练"""
    print("\n🚀 运行完整训练...")
    print("配置: 预训练 + 课程学习训练 (约4-8小时)\n")
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode train --production')


def run_train_only():
    """只训练"""
    print("\n🎯 运行训练 (跳过预训练)...")
    print("配置: 课程学习训练 (约2-4小时)\n")
    python_cmd = get_python_cmd()
    run_cmd(f'"{python_cmd}" main.py --mode train --skip_text_pretrain --skip_label_pretrain')


def run_production():
    """生产环境"""
    print("\n🏭 运行生产环境完整流程...")
    print("配置: 完整预训练 + 完整课程学习 (约6-12小时)")
    print("这将获得最佳效果，但需要较长时间\n")

    confirm = input("确认运行? (y/n): ").strip().lower()
    if confirm == 'y':
        python_cmd = get_python_cmd()
        run_cmd(f'"{python_cmd}" main.py --mode train --production')
    else:
        print("已取消")


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数"""
    print_banner()

    # 检查文件
    if not check_files():
        input("\n按Enter键退出...")
        return

    # 显示菜单并执行
    while True:
        choice = show_menu()

        if choice == '0':
            print("\n👋 再见!")
            break
        elif choice == '1':
            run_full_pipeline()
        elif choice == '2':
            run_full_quick_test()
        elif choice == '3':
            run_test_text_only()
        elif choice == '4':
            run_test_label_only()
        elif choice == '5':
            run_test_struct_only()
        elif choice == '6':
            run_full_pretrain()
        elif choice == '7':
            run_full_train()
        elif choice == '8':
            run_train_only()
        elif choice == '9':
            run_production()
        else:
            print("\n❌ 无效选项，请重新选择")
            continue

        # 询问是否继续
        continue_choice = input("\n是否继续其他操作? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 再见!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按Enter键退出...")
