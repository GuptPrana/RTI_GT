import datetime
import os

import cv2


def _init_cam(src):
    camera = cv2.VideoCapture(src)
    if not camera.isOpened():
        raise IOError("Failed to initialize camera {}".format(src))
    return camera


def _save_frames(frameL, frameR):
    # save frame with local timestamp --> hh:mm:ss.xxxxxx
    timestamp = datetime.datetime.now().time()
    filenameL = "L_" + str(timestamp).replace(":", "_").replace(".", "_") + ".jpg"
    filenameR = "R_" + str(timestamp).replace(":", "_").replace(".", "_") + ".jpg"
    filepathL = os.path.join(cwd, "data", filenameL)
    filepathR = os.path.join(cwd, "data", filenameR)
    cv2.imwrite(filepathL, frameL)
    cv2.imwrite(filepathR, frameR)
    return


if __name__ == "__main__":
    cameraL = _init_cam(0)
    cameraR = _init_cam(1)
    cwd = os.getcwd()
    frame_count = 0

    while True:
        retL, frameL = cameraL.read()
        retR, frameR = cameraR.read()

        cv2.imshow("FrameL", frameL)
        cv2.imshow("FrameR", frameR)

        # record frames at FPS=6
        if not frame_count % 6:
            _save_frames(frameL, frameR)
        frame_count += 1

        # 'q' to quit stream
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cameraL.release()
    cameraR.release()
    cv2.destroyAllWindows()
