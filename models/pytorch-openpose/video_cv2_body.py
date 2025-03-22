import os
import time

import cv2
import sys
from tqdm import tqdm
from sys import platform

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
if platform == 'win32':
    lib_dir = 'Release'
    bin_dir = 'bin'
    x64_dir = 'x64'
    lib_path = os.path.join(BASE_DIR, lib_dir)
    bin_path = os.path.join(BASE_DIR, bin_dir)
    x64_path = os.path.join(BASE_DIR, x64_dir)
    sys.path.append(lib_path)
    os.environ['PATH'] += ';' + bin_path + ';' + x64_path + '\Release;'
    try:
        import pyopenpose as op

        print("successful, import pyopenpose!")
    except ImportError as e:
        print("fail to import pyopenpose!")
        raise e
else:
    print(f"当前电脑环境:\n{platform}\n")
    sys.exit(-1)


def out_video(input):
    datum = op.Datum()
    opWrapper = op.WrapperPython()
    params = dict()
    params["model_folder"] = BASE_DIR + "\models"
    params["model_pose"] = "BODY_25"
    params["number_people_max"] = 3
    params["disable_blending"] = False
    opWrapper.configure(params)
    opWrapper.start()
    file = input.split("/")[-1]
    output = "video/out-optim-" + file
    print("It will start processing video: {}".format(input))
    cap = cv2.VideoCapture(input)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # create VideoWriter,VideoWriter_fourcc is video decode
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output, fourcc, fps, frame_size)
    # the progress bar
    with tqdm(range(frame_count)) as pbar:

        while cap.isOpened():
            start = time.time()
            success, frame = cap.read()
            if success:
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                opframe = datum.cvOutputData
                FPS = 1 / (time.time() - start)
                opframe = cv2.putText(opframe, "FPS" + str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                      (0, 255, 0), 3)
                out.write(opframe)
                pbar.update(1)
            else:
                break

    pbar.close()
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print("{} finished!".format(output))


if __name__ == "__main__":
    video_dir = "./Side_SY/1.avi"
    out_video(video_dir)
