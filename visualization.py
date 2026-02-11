"""
visualization.py - Training visualization module

Generates training/validation plots from log records.
Separated from training_logger.py for modularity.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (headless, no GUI)
import matplotlib.pyplot as plt
from typing import List, Optional


def generate_training_figures(
    records: List[dict],
    output_path: str,
    figure_dir: Optional[str] = None
) -> None:
    """
    Generate training/validation plots from log records.
    
    Args:
        records: List of log records from TrainingLogger
        output_path: Path to the logger CSV file (used for naming)
        figure_dir: Directory to save figures (default: ../figure relative to output_path)
    """
    if not records:
        print("[Visualization] No records to plot")
        return
    
    if figure_dir is None:
        figure_dir = os.path.join(os.path.dirname(output_path), "..", "figure")
    os.makedirs(figure_dir, exist_ok=True)
    
    # Extract base name from logger output path for consistent naming
    base_name = os.path.basename(output_path).replace(".csv", "")
    
    df = pd.DataFrame(records)
    
    # Skip if no epoch column
    if 'epoch' not in df.columns:
        print("[Visualization] No 'epoch' column found, skipping figure generation")
        return
    
    # Group by epoch
    epoch_df = df.groupby(df['epoch'].astype(int)).mean(numeric_only=True)
    
    # 1. Loss Plot
    _plot_loss_curve(epoch_df, figure_dir, base_name)
    
    # 2. Accuracy Plot
    _plot_accuracy_curve(epoch_df, figure_dir, base_name)
    
    # 3. Per-Head Plots (for multi-head models)
    _plot_head_curves(epoch_df, figure_dir, base_name)
    
    # 4. Per-Reward Plots (for GDPO)
    _plot_reward_curves(epoch_df, figure_dir, base_name)
    
    # Cleanup all figures to free memory
    plt.close('all')
    
    print(f"[Visualization] Figures saved to {figure_dir}")


def _plot_loss_curve(epoch_df: pd.DataFrame, figure_dir: str, base_name: str) -> None:
    """Plot Loss curve (Train vs Val) - separate image"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    has_data = False
    if 'total_loss' in epoch_df.columns:
        ax.plot(epoch_df.index, epoch_df['total_loss'], 'b-o', label='Train Loss')
        has_data = True
    if 'eval_loss' in epoch_df.columns:
        ax.plot(epoch_df.index, epoch_df['eval_loss'], 'r-o', label='Val Loss')
        has_data = True
    
    if has_data:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Validation Loss')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'loss_{base_name}.png'), dpi=150)
    plt.close()


def _plot_accuracy_curve(epoch_df: pd.DataFrame, figure_dir: str, base_name: str) -> None:
    """Plot Accuracy curve (Train vs Val) - separate image"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    has_data = False
    if 'accuracy' in epoch_df.columns:
        ax.plot(epoch_df.index, epoch_df['accuracy'], 'b-o', label='Train Acc')
        has_data = True
    if 'eval_accuracy' in epoch_df.columns:
        ax.plot(epoch_df.index, epoch_df['eval_accuracy'], 'r-o', label='Val Acc')
        has_data = True
    
    if has_data:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training vs Validation Accuracy')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'accuracy_{base_name}.png'), dpi=150)
    plt.close()


def _plot_head_curves(epoch_df: pd.DataFrame, figure_dir: str, base_name: str) -> None:
    """Plot per-head loss curves - each head as separate image"""
    head_cols = [c for c in epoch_df.columns if c.startswith('head_') and '_loss' in c]
    
    for col in head_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epoch_df.index, epoch_df[col], 'g-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(col)
        ax.set_title(f'{col} per Epoch')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'{col}_{base_name}.png'), dpi=150)
        plt.close()


def _plot_reward_curves(epoch_df: pd.DataFrame, figure_dir: str, base_name: str) -> None:
    """Plot per-reward curves - each reward as separate image"""
    reward_cols = [c for c in epoch_df.columns if c.startswith('reward_')]
    
    for col in reward_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epoch_df.index, epoch_df[col], 'g-o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(col)
        ax.set_title(f'{col} per Epoch')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'{col}_{base_name}.png'), dpi=150)
        plt.close()
