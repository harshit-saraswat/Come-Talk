# Sign Language Detector
A python based computer vision- sign language recognizer program which recognizes sign language hand gestures and give output in english language format which is understandable by most people.

### Tools Used ###
Python, Tensorflow, Keras, OpenCv
### Running this project
1. Install Python 3, Opencv 3, Tensorflow, Keras.
2. First Train the model.
    ```
    python cnn_model.py
    ```
2. Now to test the model you just need to run recognise.py . To do so just open the terminal and run following command.
    ```
    python recognise.py
    ```
    Adjust the hsv values from the track bar to segment your hand color.

3. To create your own data set.
    ```
    python capture.py
    ```
The dataset can be downloaded from: 
https://drive.google.com/open?id=1ceeZWl7EhClQRw0XevvfWPbJQIWO2uOE




