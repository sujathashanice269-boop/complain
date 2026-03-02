"""
一键启动脚本 - run.py
让代码新手可以轻松运行完整训练流程
"""

import os
import sys
# ✅ 新增：获取正确的 Python 命令
def get_python_cmd():
    """获取当前 Python 解释器路径"""
    return sys.executable
def print_banner():
    print("\n" + "="*70)
    print("        客户重复投诉预测系统 - 六个方向完全改进版")
    print("="*70)
    print("改进内容:")
    print("  ✓ 方向一: Text预训练 (30轮MLM + 20轮对比学习)")
    print("  ✓ 方向二: Label全局图预训练")
    print("  ✓ 方向三: 结构化特征重要性加权")
    print("  ✓ 方向四: 真正的跨模态注意力")
    print("  ✓ 方向五: 课程学习训练策略")
    print("  ✓ 方向六: 模态平衡损失")
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

    missing = []
    for name, file in required_files.items():
        if os.path.exists(file):
            print(f"  ✓ {name}: {file}")
        else:
            print(f"  ✗ {name}: {file} (缺失)")
            missing.append(file)

    if missing:
        print(f"\n❌ 缺少文件: {', '.join(missing)}")
        print("请确保所有必要文件都在当前目录下")
        return False

    print("\n✅ 所有必要文件齐全!\n")
    return True


def show_menu():
    """显示菜单"""
    print("\n请选择运行模式:")
    print("="*70)
    print("【快速测试模式】")
    print("1. 🚀 完整流程快速测试 (1轮预训练+1轮训练，约14小时) ← 验证整体代码!")
    print("2. 🔍 单独测试Text模型 (约47分钟)")
    print("3. 🔍 单独测试Label模型 (约3分钟) ← 推荐先测这个!")
    print("4. 🔍 单独测试Struct模型 (约3分钟)")
    print()
    print("【正式训练模式】")
    print("5. 完整预训练 (Text 30+20轮 + Label 20轮)")
    print("6. 完整训练 (预训练 + 课程学习训练)")
    print("7. 跳过预训练直接训练")
    print("8. 生产环境完整流程 (最佳效果，时间最长)")
    print()
    print("0. 退出")
    print("="*70)

    choice = input("\n请输入选项 (0-8): ").strip()
    return choice


def run_full_quick_test():
    """完整流程快速测试 - 新增功能"""
    print("\n🚀 运行完整流程快速测试...")
    print("="*70)
    print("测试内容:")
    print("  1️⃣ Text预训练阶段1 (MLM) - 1轮")
    print("  2️⃣ Text预训练阶段2 (对比学习) - 1轮")
    print("  3️⃣ Label全局图预训练 - 1轮")
    print("  4️⃣ 课程学习阶段1 (单模态) - 1轮")
    print("     • text_only")
    print("     • label_only")
    print("     • struct_only")
    print("  5️⃣ 课程学习阶段2 (双模态) - 1轮")
    print("     • text_label")
    print("     • text_struct")
    print("     • label_struct")
    print("  6️⃣ 课程学习阶段3 (三模态) - 1轮")
    print("     • full模型")
    print()
    print("⏱️  预计时间: 约14小时")
    print("💡 目的: 验证整个训练流程能否跑通")
    print("="*70)

    confirm = input("\n确认运行? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    # ✅ 修改这里
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode train --quick_test'
    print(f"\n执行命令: {cmd}\n")
    os.system(cmd)


def run_test_text_only():
    """单独测试Text"""
    print("\n🔍 单独测试Text模型...")
    print("配置: 只训练text_only模型 (约47分钟)")
    print("用途: 验证BERT预训练和文本处理是否正常\n")
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode train --quick_test --test_single_modal text'
    print(f"执行命令: {cmd}\n")
    os.system(cmd)


def run_test_label_only():
    """单独测试Label"""
    print("\n🔍 单独测试Label模型...")
    print("配置: 只训练label_only模型 (约3分钟)")
    print("用途: 快速验证GAT标签编码和全局图预训练是否正常")
    print("💡 推荐: 先测试这个，快速定位问题!\n")
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode train --quick_test --test_single_modal label'
    print(f"执行命令: {cmd}\n")
    os.system(cmd)


def run_test_struct_only():
    """单独测试Struct"""
    print("\n🔍 单独测试Struct模型...")
    print("配置: 只训练struct_only模型 (约3分钟)")
    print("用途: 验证结构化特征处理是否正常\n")
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode train --quick_test --test_single_modal struct'
    print(f"执行命令: {cmd}\n")
    os.system(cmd)


def run_full_pretrain():
    """完整预训练"""
    print("\n📚 运行完整预训练...")
    print("配置: Text(30+20轮) + Label(20轮) (约2-4小时)\n")
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode pretrain_only --production'
    print(f"执行命令: {cmd}\n")
    os.system(cmd)


def run_full_train():
    """完整训练"""
    print("\n🚀 运行完整训练...")
    print("配置: 预训练 + 课程学习训练 (约4-8小时)\n")
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode train --production'
    print(f"执行命令: {cmd}\n")
    os.system(cmd)


def run_train_only():
    """只训练"""
    print("\n🎯 运行训练 (跳过预训练)...")
    print("配置: 课程学习训练 (约2-4小时)\n")
    python_cmd = get_python_cmd()
    cmd = f'"{python_cmd}" main.py --mode train --skip_text_pretrain --skip_label_pretrain'
    print(f"执行命令: {cmd}\n")
    os.system(cmd)


def run_production():
    """生产环境"""
    print("\n🏭 运行生产环境完整流程...")
    print("配置: 完整预训练 + 完整课程学习 (约6-12小时)")
    print("这将获得最佳效果，但需要较长时间\n")

    confirm = input("确认运行? (y/n): ").strip().lower()
    if confirm == 'y':
        python_cmd = get_python_cmd()
        cmd = f'"{python_cmd}" main.py --mode train --production'
        print(f"\n执行命令: {cmd}\n")
        os.system(cmd)
    else:
        print("已取消")


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
            run_full_quick_test()  # ← 新增：完整流程快速测试
        elif choice == '2':
            run_test_text_only()
        elif choice == '3':
            run_test_label_only()
        elif choice == '4':
            run_test_struct_only()
        elif choice == '5':
            run_full_pretrain()
        elif choice == '6':
            run_full_train()
        elif choice == '7':
            run_train_only()
        elif choice == '8':
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