import argparse, yaml, torch
from tools.model_runner import ModelRunner
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=False)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = cfg['model']['type']
    runner = ModelRunner(model_type, cfg, device)
    runner.load_checkpoint(args.checkpoint)
    print("Loaded checkpoint. Use ModelRunner.enhance_and_save to run inference on images.")

if __name__ == '__main__':
    main()
