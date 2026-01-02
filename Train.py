import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import random
import math

from Loss import CombinedLoss
from Model import MainModel


class VimeoDataset(Dataset):
    def __init__(self, dataset_path, mode='train', dataset_type='triplet', crop_size=224):
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.dataset_type = dataset_type
        self.crop_size = crop_size
        
        if dataset_type == 'triplet':
            if mode == 'train':
                list_file = self.dataset_path / 'tri_trainlist.txt'
            else:
                list_file = self.dataset_path / 'tri_testlist.txt'
        else:
            if mode == 'train':
                list_file = self.dataset_path / 'sep_trainlist.txt'
            else:
                list_file = self.dataset_path / 'sep_testlist.txt'
        
        with open(list_file, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines() if line.strip() != ""]
        
        print(f"Loaded {len(self.sample_list)} samples for {mode} ({dataset_type})")
    
    def __len__(self):
        return len(self.sample_list)

    def _read_img(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img
    
    def __getitem__(self, idx):
        sample_path = self.sample_list[idx]
        folder_path = self.dataset_path / 'sequences' / sample_path
        
        if self.dataset_type == 'triplet':
            img0 = self._read_img(folder_path / 'im1.png')
            gt   = self._read_img(folder_path / 'im2.png')
            img1 = self._read_img(folder_path / 'im3.png')
        else:
            frames_idx = sorted(random.sample(range(7), 3))
            n0, n1, n2 = frames_idx
            img0 = self._read_img(folder_path / f'im{n0+1}.png')
            gt = self._read_img(folder_path / f'im{n1+1}.png')
            img1 = self._read_img(folder_path / f'im{n2+1}.png')
        
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'train':
            img0, gt, img1 = self.augment(img0, gt, img1)
        else:
            img0, gt, img1 = self.center_crop(img0, gt, img1)
        
        img0 = torch.from_numpy(img0).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        
        return img0, gt, img1
    
    def augment(self, img0, gt, img1):
        h, w = img0.shape[:2]
        
        if h > self.crop_size and w > self.crop_size:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            img0 = img0[y:y+self.crop_size, x:x+self.crop_size]
            gt = gt[y:y+self.crop_size, x:x+self.crop_size]
            img1 = img1[y:y+self.crop_size, x:x+self.crop_size]
        else:
            img0 = cv2.resize(img0, (self.crop_size, self.crop_size))
            gt = cv2.resize(gt, (self.crop_size, self.crop_size))
            img1 = cv2.resize(img1, (self.crop_size, self.crop_size))
        
        if random.random() < 0.5:
            img0 = np.flip(img0, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
            img1 = np.flip(img1, axis=1).copy()
        
        if random.random() < 0.5:
            img0 = np.flip(img0, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()
            img1 = np.flip(img1, axis=0).copy()
        
        if random.random() < 0.5:
            img0, img1 = img1, img0
        
        if random.random() < 0.5:
            rotation = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
            img0 = cv2.rotate(img0, rotation)
            gt = cv2.rotate(gt, rotation)
            img1 = cv2.rotate(img1, rotation)
        
        return img0, gt, img1
    
    def center_crop(self, img0, gt, img1):
        h, w = img0.shape[:2]
        if h > self.crop_size and w > self.crop_size:
            x = (w - self.crop_size) // 2
            y = (h - self.crop_size) // 2
            img0 = img0[y:y+self.crop_size, x:x+self.crop_size]
            gt = gt[y:y+self.crop_size, x:x+self.crop_size]
            img1 = img1[y:y+self.crop_size, x:x+self.crop_size]
        else:
            img0 = cv2.resize(img0, (self.crop_size, self.crop_size))
            gt = cv2.resize(gt, (self.crop_size, self.crop_size))
            img1 = cv2.resize(img1, (self.crop_size, self.crop_size))
        return img0, gt, img1



class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.epoch = 0 
        self.grad_clip = config.get('grad_clip', 1.0)
        
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        self.criterion = CombinedLoss(
            use_vgg=config.get('use_vgg', True),
            use_ternary=config.get('use_ternary', False)
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 1e-3)
        )
        
        self.warmup_steps = config.get('warmup_steps', 2000)
        self.total_steps = config['num_epochs'] * len(train_loader)
        
        self.best_psnr = 0.0
        self.start_epoch = 0
        self.global_step = 0

    def get_learning_rate(self, step):
        if step < self.warmup_steps:
            mul = step / self.warmup_steps
            return self.config['lr'] * mul
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            mul = 0.5 * (1 + math.cos(math.pi * progress))
            return self.config['lr'] * mul + self.config['lr_min'] * (1 - mul)

    def train_epoch(self, epoch):
        self.epoch = epoch
        self.model.train()
        
        total_loss = 0.0
        total_loss_l1 = 0.0
        total_loss_tea = 0.0
        total_loss_vgg = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1:03d} [TRAIN]", ncols=160)
        
        for batch_idx, (img0, gt, img2) in enumerate(pbar):
            img0 = img0.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)
            img2 = img2.to(self.device, non_blocking=True)
            
            lr = self.get_learning_rate(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with autocast(enabled=self.use_amp):
                pred, aux = self.model(img0, img2, gt=gt)
                
                student_weights = aux['weights_fwd_student']
                teacher_weights = aux.get('weights_fwd_teacher')
                
                loss, loss_dict = self.criterion(
                    pred=pred,
                    gt=gt,
                    merged_teacher=aux.get('merged_teacher'),
                    flow_student=student_weights,
                    flow_teacher=teacher_weights   
                )
                loss = loss / self.accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1 == len(self.train_loader)):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1

            batch_loss = loss.item() * self.accumulation_steps
            total_loss += batch_loss
            total_loss_l1 += loss_dict.get('loss_l1', 0)
            total_loss_tea += loss_dict.get('loss_tea', 0)
            total_loss_vgg += loss_dict.get('loss_vgg', 0)
            num_batches += 1

            postfix = {
                'loss': f'{batch_loss:.4f}',
                'l1': f'{loss_dict.get("loss_l1", 0):.4f}',
                'lr': f'{lr:.2e}'
            }
            if 'loss_tea' in loss_dict:
                postfix['tea'] = f'{loss_dict["loss_tea"]:.4f}'
            if 'loss_vgg' in loss_dict:
                postfix['vgg'] = f'{loss_dict["loss_vgg"]:.4f}'
            pbar.set_postfix(postfix)

        avg_loss = total_loss / num_batches
        avg_loss_l1 = total_loss_l1 / num_batches
        avg_loss_tea = total_loss_tea / num_batches
        avg_loss_vgg = total_loss_vgg / num_batches
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} | L1: {avg_loss_l1:.4f} | Tea: {avg_loss_tea:.4f} | VGG: {avg_loss_vgg:.4f}")
        
        return {
            'total': avg_loss,
            'l1': avg_loss_l1,
            'tea': avg_loss_tea,
            'vgg': avg_loss_vgg
        }

    def validate(self):
        self.model.eval()
        total_psnr = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for img0, gt, img2 in self.val_loader:
                img0 = img0.to(self.device, non_blocking=True)
                gt = gt.to(self.device, non_blocking=True)
                img2 = img2.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    pred, _ = self.model(img0, img2, gt=None)
                
                mse = F.mse_loss(pred, gt, reduction='mean')
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                
                batch_size = img0.size(0)
                total_psnr += psnr.item() * batch_size
                total_samples += batch_size
        
        final_psnr = total_psnr / total_samples if total_samples > 0 else 0
        return final_psnr

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'global_step': self.global_step
        }
        torch.save(checkpoint, '/work/checkpoints/last_checkpoint.pth')
        if is_best:
            torch.save(checkpoint, '/work/checkpoints/best_model.pth')

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.epoch = self.start_epoch
            self.best_psnr = checkpoint.get('best_psnr', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            print(f"Resumed from epoch {self.start_epoch} (Best PSNR: {self.best_psnr:.2f})")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = {
        'vimeo_path': '/work/dataset/vimeo_triplet',
        'dataset_type': 'triplet',
        'batch_size': 256,              
        'accumulation_steps': 1,        
        'num_workers': 16,              
        'lr': 1.2e-3,
        'lr_min': 1.2e-5,
        'weight_decay': 8e-4,
        'warmup_steps': 2000,
        'crop_size': 224,
        'num_epochs': 300,
        'mixed_precision': True,
        'use_vgg': True,
        'use_ternary': True,
    }

    train_ds = VimeoDataset(config['vimeo_path'], mode='train', 
                           dataset_type=config['dataset_type'],
                           crop_size=config['crop_size'])
    val_ds = VimeoDataset(config['vimeo_path'], mode='test', 
                         dataset_type=config['dataset_type'],
                         crop_size=config['crop_size'])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    model = MainModel(scales=[1,2,4,8,16,32])

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.load_checkpoint('/work/checkpoints/last_checkpoint.pth')

    for epoch in range(trainer.start_epoch, config['num_epochs']):
        train_losses = trainer.train_epoch(epoch)
        val_psnr = trainer.validate()
        
        is_best = val_psnr > trainer.best_psnr
        if is_best:
            trainer.best_psnr = val_psnr
        
        trainer.save_checkpoint(epoch, is_best=is_best)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_losses['total']:.4f} | Val PSNR: {val_psnr:.2f}dB | Best: {trainer.best_psnr:.2f}dB")
        print(f"  L1: {train_losses['l1']:.4f} | Tea: {train_losses['tea']:.4f} | VGG: {train_losses['vgg']:.4f}")

if __name__ == "__main__":
    main()