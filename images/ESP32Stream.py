import cv2

# here i have to add the ip adress of esp32 stream
url = "http://192.168.X.X:81/stream"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("ESP32 Stream", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
