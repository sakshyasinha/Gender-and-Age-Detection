import cv2

# Paths to model files (update these paths as needed)
AGE_PROTO = r'C:\Users\KIIT0001\Downloads\Gender-and-Age-Detection-master\Gender-and-Age-Detection-master\age_deploy.prototxt'
AGE_MODEL = r'C:\Users\KIIT0001\Downloads\Gender-and-Age-Detection-master\Gender-and-Age-Detection-master\age_net.caffemodel'
GENDER_PROTO = r'C:\Users\KIIT0001\Downloads\Gender-and-Age-Detection-master\Gender-and-Age-Detection-master\gender_deploy.prototxt'
GENDER_MODEL = r'C:\Users\KIIT0001\Downloads\Gender-and-Age-Detection-master\Gender-and-Age-Detection-master\gender_net.caffemodel'

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load models
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Read the image
image_path = r'C:\Users\KIIT0001\Downloads\Gender-and-Age-Detection-master\Gender-and-Age-Detection-master\kid1.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Cannot open image")
    exit()

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    face_img = image[y:y+h, x:x+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    
    # Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    
    label = f'{gender}, {age}'
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('Gender and Age Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
