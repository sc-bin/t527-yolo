import yolov5
import data_classes
import cv2
import os
import time  # 导入time模块用于计算帧率

os.environ["DISPLAY"] = ":0.0"
model_path = "model/yolov5n.nb"

# 检测图片
yolo = yolov5.YOLO5(model_path)
print(f"model: {model_path}")

# 创建全屏窗口
# cv2.namedWindow("result", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 打开摄像头并循环获取帧显示到屏幕上
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 设置为1080p
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置长度

boxes = []
frame_count = 0  # 初始化帧计数器
start_time = time.time()  # 记录开始时间
fps = 0
while True:
    # 读取一帧图像并显示出来
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    yolo.detect_async(img)
    if yolo.has_result:
        frame_count += 1
        boxes = yolo.get_result()
    # print(f"boxes: {boxes.__len__()}")

    # 到图上画框
    for box in boxes:
        label = (
            str(data_classes.class_names[box.class_index]) + " " + str(box.reliability)
        )
        (label_width, label_height), bottom = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        cv2.rectangle(
            img,
            (box.left_x, box.left_y),
            (box.right_x, box.right_y),
            (255, 255, 0),
            2,
        )
        cv2.rectangle(
            img,
            (box.left_x, box.left_y - label_height * 2),
            (box.left_x + label_width, box.left_y),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            img,
            label,
            (box.left_x, box.left_y - label_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    # 计算帧率
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:  # 每秒计算一次帧率
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    # 在图像上显示帧率
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("result", img)
    cv2.waitKey(1)
