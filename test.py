from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(0, save=True, conf=0.3, show=True,save_crop=True)

