import cv2
import mediapipe as mp
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    imageWidth, imageHeight=image.shape[:2]
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        data=[]
        for point in mp_hands.HandLandmark:
 
              normalizedLandmark = hand_landmarks.landmark[point]
              pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
 
              # print(point)
              # print(pixelCoordinatesLandmark)
              # print(normalizedLandmark)
              
              data.append(normalizedLandmark.x)
              data.append(normalizedLandmark.y)
              data.append(normalizedLandmark.z)
        print(len(data))
        out=model.predict([data])
        print(out[0])
        #image=cv2.flip(image,1)

        font = cv2.FONT_HERSHEY_SIMPLEX
  
        # org
        org = (50, 50)
  
        # fontScale
        fontScale = 1
   
        # Blue color in BGR
        color = (255, 0, 0)
  
        # Line thickness of 2 px
        thickness = 2
   
        # Using cv2.putText() method
        image = cv2.putText(image, out[0], org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()