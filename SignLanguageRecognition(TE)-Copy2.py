#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow==2.5.0 tensorflow-gpu==2.5.0 opencv-python mediapipe sklearn matplotlib')


# In[2]:


get_ipython().system('pip install pyttsx3')


# In[2]:


from easygui import *


# In[3]:


from PIL import Image, ImageTk
from itertools import count
import tkinter as tk


# In[4]:


import pyttsx3


# In[2]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# In[6]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# In[7]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results 


# In[8]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# In[9]:


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# In[11]:


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('SLR', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[12]:


results


# In[13]:


draw_landmarks(frame, results)


# In[14]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[15]:


results


# In[16]:


results.right_hand_landmarks.landmark[0].y


# In[17]:


for res in results.right_hand_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])


# In[18]:


test


# In[19]:


lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)


# In[20]:


rh


# In[21]:


lh


# In[22]:


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


# In[23]:


result_test = extract_keypoints(results)


# In[24]:


result_test


# In[25]:


np.save('0', result_test)


# In[26]:


np.load('0.npy')


# In[27]:


extract_keypoints(results).shape


# In[ ]:


21*3+21*3


# In[16]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('data') 

# Actions that we try to detect
actions = np.array([1,2,3,4,5,6,7,8,9,'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])


no_sequences = 34


sequence_length = 1200


# In[4]:


'''for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass'''


# In[ ]:


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
               #print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('SLR WEB CAM', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('SLR WEB CAM', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()


# In[17]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[18]:


label_map = {label:num for num, label in enumerate(actions)}


# In[19]:


label_map


# In[22]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[23]:


np.array(sequences).shape


# In[24]:


np.array(labels).shape


# In[25]:


X = np.array(sequences)


# In[26]:


X.shape


# In[27]:


y = to_categorical(labels).astype(int)


# In[28]:


y


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[30]:


y_test.shape


# In[50]:


y_train.shape


# In[51]:


X_test.shape


# In[52]:


X_train.shape


# In[53]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[54]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[55]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[56]:


res=[.7,0.2,0.1]


# In[57]:


actions[np.argmax(res)]


# In[58]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[59]:


model.fit(X_train, y_train, epochs=5000, callbacks=[tb_callback])


# In[ ]:


model.summary()


# In[ ]:


res = model.predict(X_test)


# In[ ]:


actions[np.argmax(res[5])]


# In[ ]:


actions[np.argmax(y_test[5])]


# In[ ]:


model.save('action.h5')


# In[ ]:


#del model


# In[ ]:


model.load_weights('action.h5')


# In[ ]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[ ]:


yhat = model.predict(X_test)


# In[ ]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[ ]:


multilabel_confusion_matrix(ytrue, yhat)


# In[ ]:


accuracy_score(ytrue, yhat)


# In[ ]:


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
     
    return output_frame


# In[ ]:


def texttospc(input_text):
    text_speech = pyttsx3.init()
    INPUT = input_text
    text_speech.say(INPUT)
    text_speech.runAndWait()
    return INPUT 


# In[ ]:



plt.imshow(prob_viz(res, actions, image, colors))


# In[ ]:


print(sentence)


# In[1]:


sequence = []
sentence = []
threshold = 0.8
cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        
        print(results)
    
        draw_styled_landmarks(image, results)
        
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-50:]
        
        if len(sequence) == 50:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:] 
                
            
        
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
 
    
        cv2.imshow('SLR CAM', image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            

texttospc(sentence) 

cap.release()
cv2.destroyAllWindows()
    

    
 


# In[ ]:


cap.release() 
cv2.destroyAllWindows()


# In[ ]:




