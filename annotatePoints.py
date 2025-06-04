import cv2


def click_event(event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, " ", y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)


if __name__ == "__main__":

    view = 1
    image = "temp.jpg"

    img = cv2.imread(f"realsense_data/cam{view}/{image}", 1)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
