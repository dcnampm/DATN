from ultralytics import YOLOv10

model = YOLOv10('/content/drive/MyDrive/DATN/data_yolov10/weights/yolov10s.pt')

model.train(data='/content/drive/MyDrive/DATN/data_yolov10/mydataset_10.yaml',
            epochs=60,
            imgsz=640,
            batch = 16,
            lr0=0.05,
            lrf=0.05,
            cos_lr=True,
            name='train_v10_',
            save_period=5,
            plots=True)