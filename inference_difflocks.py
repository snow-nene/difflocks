#!/usr/bin/env python3
import struct

# ./inference_difflocks.py \
#   --img_path=./samples/medium_11.png \
#   --out_path=./outputs_inference/

from inference.img2hair import DiffLocksInference
import subprocess
import os
import argparse
import torch
import numpy as np
import random

# Performance monitoring is optional so inference remains usable without psutil.
try:
    from utils.performance_monitor import PerformanceMonitor
    PERF_MONITORING_AVAILABLE = True
except ImportError:
    PERF_MONITORING_AVAILABLE = False

torch.manual_seed(5)
np.random.seed(5)
random.seed(5)

def run():

    #argparse
    parser = argparse.ArgumentParser(description='Run DiffLocks single-image hair inference')
    parser.add_argument('--strand_checkpoint_path', default="./checkpoints/strand_vae/strand_codec.pt", type=str, help='Path to the strandVAE checkpoint')
    parser.add_argument('--difflocks_checkpoint_path', default="./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth", type=str, help='Path to the difflocks checkpoint')
    parser.add_argument('--difflocks_config_path', default="./configs/config_scalp_texture_conditional.json", type=str, help='Path to the difflocks config')
    parser.add_argument('--rgb2mat_checkpoint_path', default="./checkpoints/rgb2material/rgb2material.pt", type=str,  help='Path to the rgb2material checkpoint')
    parser.add_argument('--blender_path', type=str, default="", help='Path to the blender executable')
    parser.add_argument('--blender_nr_threads', default=8, type=int, help='Number of threads for blender to use')
    parser.add_argument('--blender_strands_subsample', default=1.0, type=float, help='Amount of subsample of the strands(1.0=full strands, 0.5=half strands)')
    parser.add_argument('--blender_vertex_subsample', default=1.0, type=float, help='Amount of subsample of the vertices(1.0=all vertex, 0.5=half number of vertices per strand)')
    parser.add_argument('--alembic_resolution', default=7, type=int, help='Resolution of the exported alembic')
    parser.add_argument('--export_alembic', action='store_true', help='weather to export alembic or not')
    parser.add_argument('--do_shrinkwrap', action='store_true', help='applies a shrinkwrap modifier in blender that pushes the strands away from the scalp so they dont pass through the head')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to do inference on')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the image to do inference on')
    parser.add_argument('--enable_monitoring', action='store_true', help='Track inference stage time and memory')
    parser.add_argument('--slim_output', action='store_true', help='Slim output mode: skip intermediate npz/json/png saves')
    parser.add_argument('--max_num_strands', type=int, default=10000, help='Maximum number of strands in hair_binary.data')
    args = parser.parse_args()

    if args.enable_monitoring and not PERF_MONITORING_AVAILABLE:
        parser.error('Performance monitoring requires psutil; install the project dependencies.')
    if args.max_num_strands < 1:
        parser.error('--max_num_strands must be at least 1.')
    if args.slim_output and args.blender_path:
        parser.error('--slim_output omits the NPZ file required by Blender export.')

    print("args is", args)

    monitor = PerformanceMonitor() if args.enable_monitoring else None
    if monitor:
        monitor.start_pipeline()
        monitor.start_step("Model initialization")
    difflocks = DiffLocksInference(
        args.strand_checkpoint_path,
        args.difflocks_config_path,
        args.difflocks_checkpoint_path,
        args.rgb2mat_checkpoint_path,
        perf_monitor=monitor,
        slim_output=args.slim_output,
    )
    if monitor:
        monitor.end_step()

    os.makedirs(args.out_path, exist_ok=True)
    strand_points_world, hair_material_dict = difflocks.file2hair(args.img_path, args.out_path)
    if strand_points_world is None:
        parser.error('No face was detected in the input image; no hair was generated.')

    print("hair_material_dict", hair_material_dict)

    if monitor:
        monitor.start_step("Binary hair export")
    export_hair_to_binary_data(
        strand_points_world,
        os.path.join(args.out_path, "hair_binary.data"),
        max_num_strands=args.max_num_strands,
    )
    if monitor:
        monitor.end_step()

    if args.blender_path != "":
        cmd = [args.blender_path, "-t", str(args.blender_nr_threads), "--background", "--python", "./inference/npz2blender.py", "--", "--input_npz", os.path.join(args.out_path, "difflocks_output_strands.npz"), "--out_path", args.out_path, "--strands_subsample", str(args.blender_strands_subsample), "--vertex_subsample", str(args.blender_vertex_subsample), "--alembic_resolution", str(args.alembic_resolution)]
        if args.do_shrinkwrap:
            cmd.append("--shrinkwrap")
        if args.export_alembic:
            cmd.append("--export_alembic")
        if monitor:
            monitor.start_step("Blender export")
        subprocess.run(cmd, check=True)
        if monitor:
            monitor.end_step()

    print("Finished writing to ", args.out_path)

    if monitor:
        monitor.end_pipeline()
        monitor.save_report(os.path.join(args.out_path, "performance_report.txt"))
        monitor.save_json(os.path.join(args.out_path, "performance_metrics.json"))

def export_hair_to_binary_data(strand_points_world, output_path, max_num_strands=10000):
    """
    将发丝点云数据以二进制格式导出为指定.data文件，并将每个发丝的256个点采样至100个点

    参数:
        strand_points_world: torch.Tensor，形状为 [NumofStrand, 256, 3]，世界坐标系下的发丝点云
        output_path: str，输出二进制.data文件的路径（如"hair_binary.data"）
    """
    # -------------------------- 步骤1：获取发丝总数（NumofStrand） --------------------------
    if max_num_strands < 1:
        raise ValueError("max_num_strands must be at least 1")

    total_num_of_strands = strand_points_world.shape[0]
    num_of_strand = min(total_num_of_strands, max_num_strands)
    if num_of_strand < total_num_of_strands:
        strand_indices = torch.linspace(
            0, total_num_of_strands - 1, steps=num_of_strand,
            dtype=torch.long, device=strand_points_world.device,
        )
        strand_points_world = strand_points_world[strand_indices]
    print(f"发丝总数（NumofStrand）：{num_of_strand} / {total_num_of_strands}")

    # -------------------------- 步骤2：256→100点均匀采样 --------------------------
    # 生成100个均匀分布的采样索引（0到255，覆盖原始所有点，保证发丝形状）
    sample_indices = torch.linspace(0, strand_points_world.shape[1]-1, 100, dtype=torch.long)
    sampled_strands = strand_points_world[:, sample_indices, :]  # [num_of_strand, 100, 3]
    num_of_vertices_per_strand = sampled_strands.shape[1]  # 采样后每个发丝顶点数：100
    print(f"每个发丝采样后顶点数（NumOfVertices）：{num_of_vertices_per_strand}")

    # -------------------------- 步骤3：数据预处理（设备转换+numpy转换） --------------------------
    # 若张量在GPU上，先转至CPU；再转换为numpy数组（方便二进制打包）
    if strand_points_world.is_cuda:
        sampled_strands_np = sampled_strands.cpu().numpy()
    else:
        sampled_strands_np = sampled_strands.numpy()
    # 转换为float32类型（二进制存储常用，也可根据需求改为float64）
    sampled_strands_np = sampled_strands_np.astype(np.float32)

    # -------------------------- 步骤4：二进制格式写入.data文件 --------------------------
    # 使用'wb'模式打开文件（二进制写入模式，不可修改为'w'）
    with open(output_path, 'wb') as f:
        # 第一部分：写入发丝总数（NumofStrand，int类型，4字节大端/小端可配置，默认小端）
        # struct.pack格式说明：'i'=int32（4字节），'>i'=大端序，'<i'=小端序（主流系统默认小端）
        f.write(struct.pack('<i', num_of_strand))

        # 第二部分：遍历每个发丝，写入顶点数+顶点坐标（二进制格式）
        for i in range(num_of_strand):
            # 1. 写入当前发丝的顶点数（NumOfVertices，int32类型，4字节）
            f.write(struct.pack('<i', num_of_vertices_per_strand))

            # 2. 写入当前发丝的所有顶点坐标（二进制浮点型）
            current_strand_vertices = sampled_strands_np[i]  # [100, 3]
            # 扁平化顶点数据为一维数组 [300,]，直接以二进制形式写入（高效无冗余）
            # tobytes()：将numpy数组转为二进制字节流，比手动打包更高效
            f.write(current_strand_vertices.tobytes())

    print(f"二进制数据已成功导出至：{output_path}")
    print(f"二进制格式验证：")
    print(f"  - 字节偏移0-3：发丝总数（int32，{num_of_strand}）")
    print(f"  - 每个发丝起始：4字节顶点数（int32，100） + 100*3*4字节坐标（float32，300个浮点值）")
    print(f"  - 单个发丝字节数：4 + 1200 = 1204 字节")
    print(f"  - 总文件约字节数：4 + {num_of_strand} * 1204 字节")


if __name__ == '__main__':

    run()
