from ultralytics import YOLO
import cv2


def validation_model():
    model = YOLO('./eco_scan/eco_scan_weight/weights/best.pt') 
    results = model.predict('./val_img/my_test_image.jpg', save=True) 

    for result in results:
        res_plotted = result.plot()
        cv2.imshow("result", res_plotted)
        cv2.waitKey(0)

    cv2.destroyAllWindows()