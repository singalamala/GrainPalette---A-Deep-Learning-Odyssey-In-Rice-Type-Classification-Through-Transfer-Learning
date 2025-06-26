
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('rice_classification_model.h5')
class_names = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Sushi']

cap = cv2.VideoCapture(0)
IMG_SIZE = 224

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    
    cv2.putText(frame, f"Prediction: {class_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Rice Type Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
