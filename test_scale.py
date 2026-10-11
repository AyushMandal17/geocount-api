import cv2
from app import engine

m = engine.load_model("yolo26m-obb.pt")
img = cv2.imread("cars.jpg")
for imgsz in (320, 480, 640, 1024):
    for conf in (0.25, 0.5):
        d = engine.run_inference(m, img, None, conf, use_slicing=False, imgsz=imgsz)
        print(imgsz, conf, engine.count_by_class(d))
        cv2.imwrite(f"out_{imgsz}_{conf}.jpg", engine.annotate(img, d))