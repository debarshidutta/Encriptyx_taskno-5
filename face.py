import cv2
#Load the cascade
face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def face_extractor(img, margin=20):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        x1=max(0,x-margin)
        y1=max(0,y-margin)
        x2=min(x+w+margin,img.shape[1])
        y2=min(y+h+margin,img.shape[0])
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    return img
#capture image from webcam
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=face_extractor(frame)
    cv2.imshow('Face Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()