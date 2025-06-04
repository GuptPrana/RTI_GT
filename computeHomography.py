import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    view = 1
    image = 'temp.jpg'
    
    im_src = cv2.imread(f'realsense_data/cam1/{image}')
    pts_src = np.array([[269, 271], [451, 272], [217, 388], [537, 394]])

    # x = 640
    # y = 640
    # b = 120
    x = 256
    y = 256
    b = 16
    delta_x = (x-2*b)/2
    delta_y = (y-2*b)/2
    pts_dst = np.array([[0, 0],[2, 0],[0, 2],[2, 2]])
    pts_dst = delta_x * pts_dst + b

    H, _ = cv2.findHomography(pts_src, pts_dst)
    np.save(f"realsense_data/transform_{view}.npy")
    print(H)

    im_out = cv2.warpPerspective(im_src, h, (x, y))#(im_src.shape[1],im_src.shape[0]))
    cv2.imwrite('warped.jpg', im_out)
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Warped Image", im_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()