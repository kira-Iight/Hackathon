from ultralytics import YOLO
import torch

def main():
    model = YOLO('yolov8s.pt')
    
    model.train(
        data='dataset/data.yaml',   
        epochs=150,
        imgsz=640,
        batch=16,
        device='cuda',  # Явно указываем GPU
        
        # Уменьшите workers для Windows
        workers=4,  # ↓ Уменьшил с 8 до 4
        
        # Параметры для дисбаланса классов:
        cos_lr=True,
        close_mosaic=10,
        
        # Аугментации
        augment=True,
        hsv_h=0.03,
        hsv_s=0.9,
        hsv_v=0.6,
        translate=0.2,
        scale=0.8,
        fliplr=0.5,
        flipud=0.4,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.4,
        degrees=20.0,
        shear=8.0,
        perspective=0.001,
        
        # Оптимизация:
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Регуляризация:
        dropout=0.1,
        patience=30,
        project='runs/train', 
        name='tree_bush_balanced',
        exist_ok=True,
    )

if __name__ == '__main__':
    main()