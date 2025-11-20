import torch
from abc import ABC, abstractmethod
from pathlib import Path
import yaml, json
from datetime import datetime
import logging
import tempfile, shutil

class BaseTrainer(ABC):
    def __init__(self, config: dict, model: torch.nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device

        # Metrics bookkeeping
        self.metrics = {}
        self.metric_mode = self.config['training'].get('metric_mode', 'min')
        if self.metric_mode not in ('min', 'max'):
            raise ValueError("training.metric_mode must be 'min' or 'max'")
        self.best_metric = float('inf') if self.metric_mode == 'min' else -float('inf')

        # Checkpoint dir
        ck_cfg = self.config.get('logging', {})
        base_ckpt_dir = ck_cfg.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = Path(base_ckpt_dir) / self.config['model']['name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()
        self._save_config_file()

    def _setup_logging(self):
        log_dir = Path(self.config['logging'].get('log_dir', 'logs')) / self.config['model']['name']
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(self.config['model']['name'])
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(log_dir / f"train_{timestamp}.log")
            ch = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt); ch.setFormatter(fmt)
            logger.addHandler(fh); logger.addHandler(ch)
        return logger

    def _save_config_file(self):
        cfg_file = self.checkpoint_dir / 'config.yaml'
        with open(cfg_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get_additional_checkpoint_state(self):
        """Override to return extra state (optimizers, schedulers)"""
        return {}

    def load_additional_checkpoint_state(self, state: dict):
        """Load extra state if present"""
        return

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool=False):
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(self.checkpoint_dir))
        tmp_path = Path(tmp.name); tmp.close()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'extra_state': self.get_additional_checkpoint_state()
        }
        try:
            torch.save(checkpoint, tmp_path)
            latest = self.checkpoint_dir / 'latest.pth'
            shutil.move(str(tmp_path), str(latest))
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')

    def load_checkpoint(self, checkpoint_file: str) -> int:
        ck = Path(checkpoint_file)
        if not ck.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        checkpoint = torch.load(str(ck), map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            self.logger.error("Model.load_state_dict failed: %s", e)
            raise
        self.metrics = checkpoint.get('metrics', {})
        extra = checkpoint.get('extra_state', {})
        if extra:
            self.load_additional_checkpoint_state(extra)
        return int(checkpoint.get('epoch', 0)) + 1

    def update_metrics(self, metrics: dict, epoch: int):
        """Update metrics dict and save a metrics.json snapshot."""
        self.metrics.update(metrics)
        self.logger.info(f"Epoch {epoch} - " + " - ".join([f"{k}: {v:.6f}" for k,v in metrics.items()]))
        metrics_file = self.checkpoint_dir / 'metrics.json'
        tmp = metrics_file.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        tmp.replace(metrics_file)

    def _is_improved(self, metric_value: float) -> bool:
        if self.metric_mode == 'min':
            return metric_value < self.best_metric
        return metric_value > self.best_metric

    @abstractmethod
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict:
        raise NotImplementedError

    @abstractmethod
    def validate(self, dataloader: torch.utils.data.DataLoader) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        raise NotImplementedError

    def train(self, train_loader, val_loader=None, num_epochs=None):
        num_epochs = int(num_epochs or self.config['training'].get('num_epochs', self.config['training'].get('epochs', 100)))
        start_epoch = 0
        resume_path = self.config['training'].get('resume_from')
        if resume_path:
            start_epoch = self.load_checkpoint(resume_path)
        metric_name = self.config['training'].get('metric_to_monitor', 'psnr')
        save_interval = self.config['training'].get('save_interval', 5)
        
        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            train_metrics = self.train_epoch(train_loader)
            self.update_metrics({f'train_{k}': v for k, v in train_metrics.items()}, epoch)
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.update_metrics({f'val_{k}': v for k, v in val_metrics.items()}, epoch)
                
                if metric_name and metric_name in val_metrics:
                    val_val = val_metrics[metric_name]
                    is_best = self._is_improved(val_val)
                    if is_best:
                        self.best_metric = val_val
                        self.save_checkpoint(epoch, val_metrics, is_best=True)
                        self.logger.info(f"New best {metric_name}: {val_val:.6f}")
                else:
                    is_best = False
            else:
                is_best = False
                val_metrics = {}
            
            # Save checkpoint at intervals
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, val_metrics if val_loader else train_metrics, is_best=False)
        
        self.logger.info("Training completed!")
