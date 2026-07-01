from ultralytics import YOLO

model = YOLO("yolo26s-obb.pt")

results = model.train(
    data="/workspace/DOTA-v2.0/dota_v2_obb.yaml",
    
    epochs=150,
    patience=50,
    
    imgsz=1024,
    batch=32,
    workers=8,
    device=0,
    half=True,
    
    optimizer="auto",
    cos_lr=True,
    warmup_epochs=3,
    
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.15,
    degrees=10.0,
    fliplr=0.5,
    flipud=0.5,
    
    save=True,
    save_period=5,
    plots=True,
    val=True,
    verbose=True,
    
    project="/workspace/YOLO26s-DOTAv2.0/runs",
    name="yolo26s-obb-dotav2-a100",
    
    seed=42,
    deterministic=False,
)