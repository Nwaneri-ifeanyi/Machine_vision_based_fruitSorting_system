# Machine_vision_based_fruitSorting_system
Fruit Sorting Device
This is a program that uses computer vision and machine learning to sort ripe and unripe oranges. It uses an Arduino board to control a servo motor that separates the oranges based on their ripeness.

Getting Started
To run the program, you will need to do the following:

Clone the repository to your local machine
Install the necessary libraries: opencv-python, numpy, keras, pyfirmata
Train a binary classifier on the orange images using a deep learning framework such as Keras or TensorFlow.
Place the trained model file in the Training folder.
Connect your Arduino board to your machine and upload the StandardFirmata sketch to it.
Connect the servo motor to pin 9 on the Arduino board.
Run the orange_sorting.py script.
The program uses the device's webcam to capture a live video stream. It applies object detection to identify oranges in the stream and then uses the trained classifier to determine their ripeness. If an orange is identified as ripe, the servo motor moves to separate it from the unripe oranges.

Configuration
The program can be configured by changing the following parameters in the orange_sorting.py script:

thres: Threshold to detect the oranges. You can adjust this value to change the sensitivity of the object detection algorithm.
source: Source of the video stream. You can change this to use a different webcam or a video file.
classNames: List of object classes. You can add or remove classes depending on your use case.
classFile: File containing the names of the object classes.
configPath: Path to the configuration file of the object detection model.
weightsPath: Path to the weights file of the object detection model.
consecutive_frames: Number of consecutive frames a ripe orange needs to be detected before the servo motor moves. You can adjust this value to change the sensitivity of the sorting algorithm.
Contributing
If you find a bug or want to suggest an improvement, feel free to open an issue or submit a pull request.
