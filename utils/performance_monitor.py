#!/usr/bin/env python3
"""
Performance monitoring utility for ETG-Hair inference pipeline
Tracks GPU/CPU memory usage and execution time for each pipeline step
"""

import time
import torch
import psutil
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


class PerformanceMonitor:
    """Monitor and track performance metrics throughout the inference pipeline"""

    def __init__(self, enable_gpu=True, enable_cpu=True):
        """
        Initialize performance monitor

        Args:
            enable_gpu: Whether to track GPU memory
            enable_cpu: Whether to track CPU memory
        """
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.enable_cpu = enable_cpu
        self.steps: List[Dict] = []
        self.current_step = None
        self.start_time = None
        self.pipeline_start_time = None

    def start_pipeline(self):
        """Mark the start of the entire pipeline"""
        self.pipeline_start_time = time.time()
        self.steps = []

    def end_pipeline(self) -> float:
        """Mark the end of the pipeline and return total time"""
        if self.pipeline_start_time is None:
            return 0.0
        return time.time() - self.pipeline_start_time

    def start_step(self, step_name: str):
        """Start tracking a new step"""
        if self.current_step is not None:
            self.end_step()

        if self.enable_gpu:
            torch.cuda.reset_peak_memory_stats()

        self.current_step = {
            'name': step_name,
            'start_time': time.time(),
            'gpu_mem_start': self._get_gpu_memory() if self.enable_gpu else None,
            'cpu_mem_start': self._get_cpu_memory() if self.enable_cpu else None,
            'gpu_mem_peak': None,
            'cpu_mem_peak': None,
        }

    def end_step(self) -> Optional[Dict]:
        """End tracking the current step and return metrics"""
        if self.current_step is None:
            return None

        end_time = time.time()
        current_step = self.current_step

        current_step['elapsed_time'] = end_time - current_step['start_time']
        current_step['gpu_mem_end'] = self._get_gpu_memory() if self.enable_gpu else None
        current_step['cpu_mem_end'] = self._get_cpu_memory() if self.enable_cpu else None

        if self.enable_gpu:
            current_step['gpu_mem_peak'] = max(
                current_step.get('gpu_mem_peak') or 0,
                torch.cuda.max_memory_allocated() / (1024 ** 3),
            )

        # If peak memory was not updated during the step (to avoid JIT/LLVM errors),
        # estimate it using the max of start and end memory
        if current_step['gpu_mem_peak'] is None and self.enable_gpu:
            start_mem = current_step.get('gpu_mem_start') or 0
            end_mem = current_step.get('gpu_mem_end') or 0
            current_step['gpu_mem_peak'] = max(start_mem, end_mem)

        if current_step['cpu_mem_peak'] is None and self.enable_cpu:
            start_mem = current_step.get('cpu_mem_start') or 0
            end_mem = current_step.get('cpu_mem_end') or 0
            current_step['cpu_mem_peak'] = max(start_mem, end_mem)

        # Calculate allocated memory
        if current_step['gpu_mem_start'] is not None and current_step['gpu_mem_end'] is not None:
            current_step['gpu_mem_allocated'] = max(0, current_step['gpu_mem_end'] - current_step['gpu_mem_start'])
        else:
            current_step['gpu_mem_allocated'] = None

        if current_step['cpu_mem_start'] is not None and current_step['cpu_mem_end'] is not None:
            current_step['cpu_mem_allocated'] = max(0, current_step['cpu_mem_end'] - current_step['cpu_mem_start'])
        else:
            current_step['cpu_mem_allocated'] = None

        self.steps.append(current_step)
        self.current_step = None
        return current_step

    def update_gpu_peak(self):
        """Update GPU peak memory for current step"""
        if self.current_step is None or not self.enable_gpu:
            return
        gpu_mem = self._get_gpu_memory()
        if self.current_step['gpu_mem_peak'] is None:
            self.current_step['gpu_mem_peak'] = gpu_mem
        else:
            self.current_step['gpu_mem_peak'] = max(self.current_step['gpu_mem_peak'], gpu_mem)

    def update_cpu_peak(self):
        """Update CPU peak memory for current step"""
        if self.current_step is None or not self.enable_cpu:
            return
        cpu_mem = self._get_cpu_memory()
        if self.current_step['cpu_mem_peak'] is None:
            self.current_step['cpu_mem_peak'] = cpu_mem
        else:
            self.current_step['cpu_mem_peak'] = max(self.current_step['cpu_mem_peak'], cpu_mem)

    def update_peak_memory(self):
        """Update peak memory for current step"""
        self.update_gpu_peak()
        self.update_cpu_peak()

    @staticmethod
    def _get_gpu_memory() -> float:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    @staticmethod
    def _get_cpu_memory() -> float:
        """Get current CPU memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.steps:
            return {}

        times = [s['elapsed_time'] for s in self.steps]
        gpu_mems = [s['gpu_mem_peak'] or 0 for s in self.steps if s['gpu_mem_peak'] is not None]
        cpu_mems = [s['cpu_mem_peak'] or 0 for s in self.steps if s['cpu_mem_peak'] is not None]

        return {
            'total_steps': len(self.steps),
            'total_time': sum(times),
            'avg_step_time': sum(times) / len(times) if times else 0,
            'max_step_time': max(times) if times else 0,
            'min_step_time': min(times) if times else 0,
            'peak_gpu_memory': max(gpu_mems) if gpu_mems else 0,
            'avg_gpu_memory': sum(gpu_mems) / len(gpu_mems) if gpu_mems else 0,
            'peak_cpu_memory': max(cpu_mems) if cpu_mems else 0,
            'avg_cpu_memory': sum(cpu_mems) / len(cpu_mems) if cpu_mems else 0,
        }

    def print_report(self) -> str:
        """Generate and print a formatted performance report"""
        total_time = self.end_pipeline()

        report = []
        report.append("=" * 80)
        report.append("ETG-Hair Performance Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Pipeline Time: {total_time:.2f}s")
        report.append("-" * 80)
        report.append("Step-wise Performance Metrics")
        report.append("-" * 80)

        # Header
        header = f"{'Step Name':<40} {'Time (s)':>12} {'GPU Mem (GB)':>15} {'CPU Mem (GB)':>15}"
        report.append(header)
        report.append("-" * 80)

        # Steps
        for step in self.steps:
            name = step['name'][:38]
            time_val = f"{step['elapsed_time']:.3f}"
            gpu_val = f"{step['gpu_mem_peak'] or 0:.2f}" if step['gpu_mem_peak'] is not None else "N/A"
            cpu_val = f"{step['cpu_mem_peak'] or 0:.2f}" if step['cpu_mem_peak'] is not None else "N/A"

            line = f"{name:<40} {time_val:>12} {gpu_val:>15} {cpu_val:>15}"
            report.append(line)

        report.append("-" * 80)

        # Total row
        total_gpu = max([s['gpu_mem_peak'] or 0 for s in self.steps]) if self.steps else 0
        total_cpu = max([s['cpu_mem_peak'] or 0 for s in self.steps]) if self.steps else 0
        total_time_val = f"{total_time:.3f}"
        line = f"{'TOTAL':<40} {total_time_val:>12} {total_gpu:>15.2f} {total_cpu:>15.2f}"
        report.append(line)
        report.append("-" * 80)

        # Detailed step metrics
        report.append("Detailed Step Metrics")
        report.append("-" * 80)

        for step in self.steps:
            report.append(f"Step: {step['name']}")
            report.append(f"  Execution Time:         {step['elapsed_time']:.3f}s")

            if self.enable_gpu:
                report.append("  GPU Memory:")
                if step['gpu_mem_start'] is not None:
                    report.append(f"    - Start:              {step['gpu_mem_start']:.2f} GB")
                if step['gpu_mem_end'] is not None:
                    report.append(f"    - End:                {step['gpu_mem_end']:.2f} GB")
                if step['gpu_mem_peak'] is not None:
                    report.append(f"    - Peak:               {step['gpu_mem_peak']:.2f} GB")
                if step['gpu_mem_allocated'] is not None:
                    report.append(f"    - Allocated:          {step['gpu_mem_allocated']:.2f} GB")

            if self.enable_cpu:
                report.append("  CPU Memory:")
                if step['cpu_mem_start'] is not None:
                    report.append(f"    - Start:              {step['cpu_mem_start']:.2f} GB")
                if step['cpu_mem_end'] is not None:
                    report.append(f"    - End:                {step['cpu_mem_end']:.2f} GB")
                if step['cpu_mem_allocated'] is not None:
                    report.append(f"    - Allocated:          {step['cpu_mem_allocated']:.2f} GB")

            report.append("")

        # Summary statistics
        summary = self.get_summary()
        report.append("-" * 80)
        report.append("Summary Statistics")
        report.append("-" * 80)
        report.append(f"Total Steps:                {summary.get('total_steps', 0)}")
        report.append(f"Average Step Time:         {summary.get('avg_step_time', 0):.3f}s")
        report.append(f"Max Step Time:             {summary.get('max_step_time', 0):.3f}s")
        report.append(f"Min Step Time:             {summary.get('min_step_time', 0):.3f}s")
        report.append(f"Peak GPU Memory (Overall):  {summary.get('peak_gpu_memory', 0):.2f} GB")
        report.append(f"Avg GPU Memory (Overall):   {summary.get('avg_gpu_memory', 0):.2f} GB")
        report.append(f"Peak CPU Memory (Overall):  {summary.get('peak_cpu_memory', 0):.2f} GB")
        report.append(f"Avg CPU Memory (Overall):   {summary.get('avg_cpu_memory', 0):.2f} GB")
        report.append("=" * 80)

        report_str = "\n".join(report)
        print(report_str)

        return report_str

    def save_report(self, file_path: str):
        """Save performance report to file"""
        report = self.print_report()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nPerformance report saved to: {file_path}")

    def save_json(self, file_path: str):
        """Save detailed metrics as JSON"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_pipeline_time': self.end_pipeline(),
            'steps': self.steps,
            'summary': self.get_summary(),
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Performance metrics saved to: {file_path}")
