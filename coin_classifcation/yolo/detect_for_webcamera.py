import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


coin_labels = {
    0: {"label": "one", "amount": 1},
    1: {"label": "five", "amount": 5},
    2: {"label": "ten", "amount": 10},
    3: {"label": "fifty", "amount": 50},
    4: {"label": "one_hundred", "amount": 100},
    5: {"label": "five_hundred", "amount": 500},
}


def detect(source, weights, device, imgsz, iou_thres, conf_thres):
    webcam = source.isnumeric() 

    # Initialize
    set_logging()
    device = select_device("0")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors = [(51,102,255), (255,255,0), (153,51,0), (0,255,0), (255,153,0), (0,0,255)]


    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            total_amount = 0
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    #label = names[int(cls)]
                    amount = coin_labels[int(cls)]["amount"]
                    total_amount += amount
                    #label_with_amount = f'{label} {conf:.2f}_yen ({amount}_yen)'
                    
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                cv2.putText(im0, f'{total_amount}_yen', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (225, 255, 255), thickness=2)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        cv2.imshow(str(p), im0)
        if cv2.waitKey(1) == ord('c'):  # 'q'キーが押されたら終了
            cv2.destroyAllWindows()
            exit()

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    with torch.no_grad():
        detect("0", "C:\\Users\\nanan\\OneDrive\\document\\GitHub\\memo\\yolov7\\runs\\train\\dataset_alllabel2\\weights\\best.pt", device, imgsz=640, iou_thres=0.4,conf_thres=0.5)