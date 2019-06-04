import cv2
import numpy as np

imgCounter = 0

def nothing(x):
    pass
#url="http://10.1.240.52:8080/shot.jpg"
image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')



def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       global imgCounter
       if result[0][0] == 1:
              imgCounter+=1
              return 'A'
       elif result[0][1] == 1:
              imgCounter+=1
              return 'B'
       elif result[0][2] == 1:
              imgCounter+=1
              return 'C'
       elif result[0][3] == 1:
              imgCounter+=1
              return 'D'
       elif result[0][4] == 1:
              imgCounter+=1
              return 'E'
       elif result[0][5] == 1:
              imgCounter+=1
              return 'F'
       elif result[0][6] == 1:
              imgCounter+=1
              return 'G'
       elif result[0][7] == 1:
              imgCounter+=1
              return 'H'
       elif result[0][8] == 1:
              imgCounter+=1
              return 'I'
       elif result[0][9] == 1:
              imgCounter+=1
              return 'J'
       elif result[0][10] == 1:
              imgCounter+=1
              return 'K'
       elif result[0][11] == 1:
              imgCounter+=1
              return 'L'
       elif result[0][12] == 1:
              imgCounter+=1
              return 'M'
       elif result[0][13] == 1:
              imgCounter+=1
              return 'N'
       elif result[0][14] == 1:
              imgCounter+=1
              return 'O'
       elif result[0][15] == 1:
              imgCounter+=1
              return 'P'
       elif result[0][16] == 1:
              imgCounter+=1
              return 'Q'
       elif result[0][17] == 1:
              imgCounter+=1
              return 'R'
       elif result[0][18] == 1:
              imgCounter+=1
              return 'S'
       elif result[0][19] == 1:
              imgCounter+=1
              return 'T'
       elif result[0][20] == 1:
              imgCounter+=1
              return 'U'
       elif result[0][21] == 1:
              imgCounter+=1
              return 'V'
       elif result[0][22] == 1:
              imgCounter+=1
              return 'W'
       elif result[0][23] == 1:
              imgCounter+=1
              return 'X'
       elif result[0][24] == 1:
              imgCounter+=1
              return 'Y'
       elif result[0][25] == 1:
              imgCounter+=1
              return 'Z'
       
cam = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars",cv2.WINDOW_NORMAL)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 26, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 10, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.namedWindow("test")

string=""
img_text = ''

while True:    
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425,100),(625,300), (0,0,255), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    
    img_text = predictor()
    if(imgCounter>=30):
        cv2.putText(frame, img_text, (80, 400), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255))
        string+= img_text
        #cv2.putText(frame, string, (40, 200), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255))
        #print(string)
        imgCounter=0
    cv2.putText(frame, img_text, (80, 400), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255))
    #cv2.putText(frame, string, (40, 200), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255)) 
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    
    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()