"""
Evaluation metrics for Astiron Spectra
Handles performance evaluation, ROC curves, and anomaly detection metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score
)
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_anomaly_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                            y_pred: Optional[np.ndarray] = None,
                            threshold: Optional[float] = None) -> Dict:
    """
    Calculate comprehensive anomaly detection metrics
    
    Args:
        y_true: Ground truth binary labels (1D array)
        y_scores: Anomaly scores (1D array)
        y_pred: Binary predictions (optional, computed from scores if not provided)
        threshold: Threshold for binary classification (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_scores = y_scores.flatten()
    
    # Remove invalid values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_scores) | np.isinf(y_scores))
    y_true = y_true[valid_mask]
    y_scores = y_scores[valid_mask]
    
    if len(y_true) == 0:
        return {'error': 'No valid data points'}
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return {'error': 'Only one class present in ground truth'}
    
    metrics = {}
    
    # ROC AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    except Exception as e:
        metrics['roc_auc'] = None
        metrics['roc_auc_error'] = str(e)
    
    # Precision-Recall AUC
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        metrics['pr_auc'] = auc(recall, precision)
        metrics['average_precision'] = average_precision_score(y_true, y_scores)
    except Exception as e:
        metrics['pr_auc'] = None
        metrics['average_precision'] = None
        metrics['pr_error'] = str(e)
    
    # Binary classification metrics
    if y_pred is None:
        if threshold is None:
            # Use Otsu's threshold or median
            threshold = compute_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
    
    if y_pred is not None:
        y_pred = y_pred.flatten()[valid_mask]
        
        metrics['threshold'] = threshold
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Data statistics
    metrics['n_samples'] = len(y_true)
    metrics['n_anomalies'] = int(np.sum(y_true))
    metrics['anomaly_rate'] = float(np.mean(y_true))
    
    return metrics


def compute_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                            method: str = 'f1') -> float:
    """
    Compute optimal threshold for binary classification
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Anomaly scores
        method: Method for threshold selection ('f1', 'youden', 'precision_recall')
        
    Returns:
        Optimal threshold value
    """
    if method == 'f1':
        # Find threshold that maximizes F1 score
        thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
        best_f1 = 0
        best_threshold = np.median(y_scores)
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    elif method == 'youden':
        # Youden's J statistic (sensitivity + specificity - 1)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    
    elif method == 'precision_recall':
        # Balance precision and recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx] if best_idx < len(thresholds) else np.median(y_scores)
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def calculate_spatial_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            pixel_area_m2: float = 900) -> Dict:
    """
    Calculate spatial metrics for anomaly detection
    
    Args:
        y_true: Ground truth binary mask (2D array)
        y_pred: Predicted binary mask (2D array)
        pixel_area_m2: Area of each pixel in square meters
        
    Returns:
        Dictionary of spatial metrics
    """
    from scipy import ndimage
    
    metrics = {}
    
    # Basic area calculations
    true_area_pixels = np.sum(y_true)
    pred_area_pixels = np.sum(y_pred)
    
    metrics['true_area_m2'] = float(true_area_pixels * pixel_area_m2)
    metrics['pred_area_m2'] = float(pred_area_pixels * pixel_area_m2)
    metrics['area_error_m2'] = float((pred_area_pixels - true_area_pixels) * pixel_area_m2)
    metrics['area_error_percent'] = float(metrics['area_error_m2'] / metrics['true_area_m2'] * 100) if metrics['true_area_m2'] > 0 else 0
    
    # Connected component analysis
    true_labels, true_n_components = ndimage.label(y_true)
    pred_labels, pred_n_components = ndimage.label(y_pred)
    
    metrics['true_n_objects'] = int(true_n_components)
    metrics['pred_n_objects'] = int(pred_n_components)
    
    # Object size statistics
    if true_n_components > 0:
        true_sizes = [np.sum(true_labels == i) for i in range(1, true_n_components + 1)]
        metrics['true_mean_object_size_m2'] = float(np.mean(true_sizes) * pixel_area_m2)
        metrics['true_max_object_size_m2'] = float(np.max(true_sizes) * pixel_area_m2)
    
    if pred_n_components > 0:
        pred_sizes = [np.sum(pred_labels == i) for i in range(1, pred_n_components + 1)]
        metrics['pred_mean_object_size_m2'] = float(np.mean(pred_sizes) * pixel_area_m2)
        metrics['pred_max_object_size_m2'] = float(np.max(pred_sizes) * pixel_area_m2)
    
    # Intersection over Union (IoU)
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    metrics['iou'] = float(intersection / union) if union > 0 else 0
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                  output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Anomaly scores
        output_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray,
                               output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Anomaly scores
        output_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f}, AP = {avg_precision:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
               label=f'Random classifier (AP = {baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_metrics_report(metrics: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate a formatted metrics report
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save report (optional)
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "Astiron Spectra - Anomaly Detection Metrics Report",
        "=" * 50,
        ""
    ]
    
    # Performance metrics
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        report_lines.extend([
            "Performance Metrics:",
            f"  ROC AUC:           {metrics['roc_auc']:.4f}",
            f"  PR AUC:            {metrics.get('pr_auc', 'N/A'):.4f}" if metrics.get('pr_auc') else "  PR AUC:            N/A",
            f"  Average Precision: {metrics.get('average_precision', 'N/A'):.4f}" if metrics.get('average_precision') else "  Average Precision: N/A",
            ""
        ])
    
    # Classification metrics
    if 'f1' in metrics:
        report_lines.extend([
            "Classification Metrics:",
            f"  F1 Score:          {metrics['f1']:.4f}",
            f"  Precision:         {metrics['precision']:.4f}",
            f"  Recall:            {metrics['recall']:.4f}",
            f"  Specificity:       {metrics.get('specificity', 'N/A'):.4f}" if 'specificity' in metrics else "  Specificity:       N/A",
            f"  Threshold:         {metrics.get('threshold', 'N/A'):.4f}" if 'threshold' in metrics else "  Threshold:         N/A",
            ""
        ])
    
    # Confusion matrix
    if 'true_positives' in metrics:
        report_lines.extend([
            "Confusion Matrix:",
            f"  True Positives:    {metrics['true_positives']}",
            f"  False Positives:   {metrics['false_positives']}",
            f"  True Negatives:    {metrics['true_negatives']}",
            f"  False Negatives:   {metrics['false_negatives']}",
            ""
        ])
    
    # Data statistics
    if 'n_samples' in metrics:
        report_lines.extend([
            "Data Statistics:",
            f"  Total Samples:     {metrics['n_samples']}",
            f"  Anomalies:         {metrics['n_anomalies']}",
            f"  Anomaly Rate:      {metrics['anomaly_rate']:.4f}",
            ""
        ])
    
    # Spatial metrics
    if 'true_area_m2' in metrics:
        report_lines.extend([
            "Spatial Metrics:",
            f"  True Area:         {metrics['true_area_m2']:.1f} m²",
            f"  Predicted Area:    {metrics['pred_area_m2']:.1f} m²",
            f"  Area Error:        {metrics['area_error_m2']:.1f} m² ({metrics['area_error_percent']:.1f}%)",
            f"  IoU:               {metrics.get('iou', 'N/A'):.4f}" if 'iou' in metrics else "  IoU:               N/A",
            ""
        ])
    
    # Errors
    errors = [key for key in metrics.keys() if key.endswith('_error')]
    if errors:
        report_lines.extend(["Errors:"])
        for error_key in errors:
            report_lines.append(f"  {error_key}: {metrics[error_key]}")
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report