#!/usr/bin/env python3
"""
Complete workflow: Train all models, visualize, and calculate metrics.
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_all import train_model, MODELS
from scripts.visualize_data import main as visualize_data
from scripts.visualize_metrics import main as visualize_metrics
from scripts.compare_model_outputs import main as compare_outputs

def main():
    """Complete workflow."""
    print("="*70)
    print("COMPLETE MODEL TRAINING AND EVALUATION WORKFLOW")
    print("="*70)
    print()
    
    # Step 1: Train all models
    print("STEP 1: Training all models...")
    print("-"*70)
    for config_path, model_name in MODELS:
        try:
            print(f"\nTraining {model_name}...")
            train_model(config_path, model_name)
            print(f"✓ {model_name} training completed")
        except Exception as e:
            print(f"✗ Error training {model_name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("STEP 2: Visualizing training data...")
    print("-"*70)
    try:
        visualize_data()
        print("✓ Data visualization completed")
    except Exception as e:
        print(f"✗ Error in data visualization: {e}")
    
    print("\n" + "="*70)
    print("STEP 3: Visualizing training metrics...")
    print("-"*70)
    try:
        visualize_metrics()
        print("✓ Metrics visualization completed")
    except Exception as e:
        print(f"✗ Error in metrics visualization: {e}")
    
    print("\n" + "="*70)
    print("STEP 4: Comparing model outputs and accuracy...")
    print("-"*70)
    try:
        compare_outputs()
        print("✓ Model comparison completed")
    except Exception as e:
        print(f"✗ Error in model comparison: {e}")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - Data visualizations: data_visualization_*.png")
    print("  - Training metrics: metrics_training_*.png")
    print("  - Model outputs: metrics_output_*.png")
    print("  - Accuracy comparisons: accuracy_comparison_*.png")
    print("\nNext step: Deploy the web app!")

if __name__ == '__main__':
    main()

