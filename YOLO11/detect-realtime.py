import cv2
from ultralytics import solutions


def workouts_live(model_path, point_list, up_angle=130.0, down_angle=90.0, show=True, save=False):
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果有多个摄像头可尝试 1、2

    assert cap.isOpened(), "Error: Unable to open camera."

    if save:
        w, h, fps = (int(cap.get(x)) for x in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out_path = "./runs/live_output.avi"
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30, (w, h))
    else:
        video_writer = None

    gym = solutions.AIGym(
        model=model_path,
        show=show,
        line_width=2,
        up_angle=up_angle,
        down_angle=down_angle,
        kpts=point_list
    )

    print("[INFO] Starting live detection. Press 'q' to quit.")
    while True:
        success, im0 = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        im0 = gym.monitor(im0)

        if save and video_writer:
            video_writer.write(im0)

        cv2.imshow("Live Detection", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = "./weights/yolo11x-pose.pt"
    point_list = [6, 8, 10]  # 头为正的引体向上，检测右肩、右肘、右手三个点形成的夹角。
    workouts_live(model_path, point_list, save=False)
