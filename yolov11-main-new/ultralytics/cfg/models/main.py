from linecache import cache
import warnings

from huggingface_hub.cli.inference_endpoints import resume

from ultralytics import YOLO, RTDETR

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    # 1. 初始化模型
    model = RTDETR(r'/root/autodl-tmp/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-x.yaml')

    # 2. 训练模型
    results = model.train(
        data=r'/root/autodl-tmp/dataset_final_v3_1_fixed/data.yaml ',
        epochs=250,
        batch=16,
        imgsz=640,
        device=0,
        patience=60,
        workers=0,  # ✅ 【最关键】多进程数据加载，Windows下建议4~8
        cache='ram',  # ✅ 【关键】将图片缓存到内存，避免重复读盘
        amp=True,  # ✅ 【关键】混合精度训练，减少显存+加速计算
        pretrained=False,
        project="runs",
        name="rtdetr-x_seed123",
        optimizer='SGD',
        resume=False,
        seed=123,
    )
    # 3. 模型验证
    metrics = model.val()