#!/usr/bin/env python3
"""
Complete automated workflow: Wait for training, visualize, get metrics, and prepare for deployment.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_all_checkpoints_exist():
    """Check if all model checkpoints exist."""
    checkpoints = [
        'checkpoints/srgan_model/best.pth',
        'checkpoints/noise2noise_model/best.pth',
        'checkpoints/deblurgan_model/best.pth'
    ]
    return all(Path(cp).exists() for cp in checkpoints)

def wait_for_training(max_wait_minutes=120, check_interval=60):
    """Wait for training to complete."""
    print("="*70)
    print("WAITING FOR TRAINING TO COMPLETE")
    print("="*70)
    print(f"Checking every {check_interval} seconds...")
    print(f"Maximum wait time: {max_wait_minutes} minutes")
    print()
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while True:
        elapsed = time.time() - start_time
        
        if check_all_checkpoints_exist():
            print("\n✓ All checkpoints found! Training complete.")
            return True
        
        if elapsed > max_wait_seconds:
            print(f"\n⚠ Maximum wait time reached ({max_wait_minutes} minutes)")
            print("Some models may still be training.")
            return False
        
        # Check individual models
        models = ['srgan', 'noise2noise', 'deblurgan']
        status = []
        for model in models:
            cp = Path(f'checkpoints/{model}_model/best.pth')
            status.append('✓' if cp.exists() else '⏳')
        
        elapsed_min = int(elapsed / 60)
        elapsed_sec = int(elapsed % 60)
        print(f"[{elapsed_min:02d}:{elapsed_sec:02d}] Status: {' '.join(status)} - Waiting...", end='\r')
        
        time.sleep(check_interval)

def main():
    """Complete workflow."""
    print("="*70)
    print("COMPLETE WORKFLOW: TRAIN → VISUALIZE → METRICS → DEPLOY")
    print("="*70)
    print()
    
    # Step 1: Wait for training
    training_complete = wait_for_training(max_wait_minutes=120)
    
    if not training_complete:
        print("\n⚠ Some models may not be fully trained.")
        print("You can continue with available models or wait longer.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Run this script again when training completes.")
            return
    
    # Step 2: Visualize data
    print("\n" + "="*70)
    print("STEP 2: VISUALIZING TRAINING DATA")
    print("="*70)
    try:
        from scripts.visualize_data import main as viz_data
        viz_data()
        print("✓ Data visualization completed")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Step 3: Visualize metrics
    print("\n" + "="*70)
    print("STEP 3: VISUALIZING TRAINING METRICS")
    print("="*70)
    try:
        from scripts.visualize_metrics import main as viz_metrics
        viz_metrics()
        print("✓ Metrics visualization completed")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Step 4: Compare outputs
    print("\n" + "="*70)
    print("STEP 4: COMPARING MODEL OUTPUTS AND ACCURACY")
    print("="*70)
    try:
        from scripts.compare_model_outputs import main as compare
        compare()
        print("✓ Model comparison completed")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Step 5: Prepare deployment
    print("\n" + "="*70)
    print("STEP 5: PREPARING FOR DEPLOYMENT")
    print("="*70)
    
    # Check if web app is ready
    webapp_file = Path('webui/streamlit_app.py')
    if webapp_file.exists():
        print("✓ Web app ready")
    else:
        print("✗ Web app not found")
    
    # Check Docker files
    dockerfile = Path('Dockerfile')
    docker_compose = Path('docker-compose.yml')
    
    if dockerfile.exists() and docker_compose.exists():
        print("✓ Docker configuration ready")
    else:
        print("✗ Docker configuration missing")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review generated visualizations")
    print("  2. Deploy web app:")
    print("     - Docker: docker-compose up -d")
    print("     - Direct: streamlit run webui/streamlit_app.py")
    print("  3. Access at: http://localhost:8501")
    print()

if __name__ == '__main__':
    main()

