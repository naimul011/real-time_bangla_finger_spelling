import argparse
import time
from pathlib import Path

import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def detect(source, weights, img_size):
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())

    if source.isnumeric():
        source = int(source)
    else:
        source = str(source)
    cap = cv2.VideoCapture(source)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

    # Create text file to save detected classes
    txt_file = open("detected_classes.txt", "w")

    names = model.module.names if hasattr(model, 'module') else model.names
    while cap.isOpened():
        ret, img0 = cap.read()
        if not ret:
            break

        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Predict
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=3)

                    # Write detected classes to text file
                    txt_file.write(f"{names[c]}\n")

        # Save the frame and display
        out.write(img0)
        cv2.imshow('Real-time Object Detection', img0)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    txt_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    args = parser.parse_args()

    detect(args.source, args.weights, args.img_size)
