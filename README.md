## Introduction

Sign language is a vital means of communication for people with hearing disabilities. This project leverages computer vision and machine learning to interpret hand gestures from a video stream, thereby bridging the communication gap between the hearing impaired and others.

Key features of this project include:
- Detection and recognition of ASL gestures in real-time.
- Utilization of deep learning models for accurate classification.
- Customizable for additional gesture recognition beyond ASL.

## Installation
## DEMO
[![Watch the video](https://github.com/ShimaaAbdalraheem/sign-language-detector-python-master/issues/1#issue-2233747335)]


### Clone the repository

```bash
git clone https://github.com/computervisioneng/sign-language-detector-python.git
```
<br>

1. Navigate to the project directory:

```bash
cd sign-language-detector-python
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

<br> 

## Usage

## IMAGE COLLECTING:
 
 Here is a file to collect Data
-to collect images: 
1- run 'collect_imgs.py'
2- after the camera is opened press k to start collecting images ,perform the sign, move you hand in many posisions as it will take a (dataset_size) of images automatically 
3- then press k again to collect images of another sign(class) or press Esc to quit
<br> 
-------------------------------------------------------------------------------------------------

## DATA PROCESSING:

- after finishing collecting all the photos , run the file 'create_dataset.py' to start making frames of each class and extract landmarks and store them 

-------------------------------------------------------------------------------------------------

## TRAINING

- Run the file 'train_classifier.py' to start training the model , after finishing , the accuracy percentage will be printed as
' percentage % of the samples were classified correctly ' and a trained model will be created and saved in 'model.p'

-------------------------------------------------------------------------------------------------

## TESTING AND USING 

-first enter the labels you want in the file 'inference_classifier.py'  in the order of classes like this 
```python
labels_dict = {0: 'peace', 1: 'yes', 2: 'thumbs up', 3: 'Hi', 4: 'No', 5: 'thank you', 6: 'sorry', 7: 'Close',8: 'Blind',9:'Phone',
               10:'Horse',11:'Hearing',12:'Soon',13:'Self',14:'Fireman',15:'Talk'
              } ```
<br> 
-run the file 'inference_classifier.py' to try the model
-------------------------------------------------------------------------------------------------
<br>
## EVALUATION

-run the file 'evaluate.py' to evalute model performance and calculate accurecy also you will get the confusion_matrix  and the result will be saved in 'performance_analysis.txt'
<br>
-------------------------------------------------------------------------------------------------

## Project Structure 

data/: Folder containing files for each class, numbered starting from 0.
collect_imgs.py: Script to open the webcam for data collection.
last_class.txt: script that saves the number of the the last class you enter automatically.
create_dataset.py: Script to create the dataset by extracting landmarks from collected images.
data.pickle: File created after running create_dataset.py wiches saves extracted Handlandmarks information.
inference_classifier.py: File containing the label for each class.
model.p: Pickle file generated after running train_classifier.py, containing the saved model.
evaluate.py: script to evalute the model and get accurecy,performance and confusion_matrix.
performance_analysis.txt: script tha saves evaluate.py performance and accurecy result. 
-------------------------------------------------------------------------------------------------
## License
This project is licensed under the MIT License. See the LICENSE file for details.


This README file includes sections such as Introduction, Installation, Usage, Contributing, and License, providing users with essential information about the project and how to get started with it. Additionally, it includes placeholders for relevant images (such as demo.gif) and code snippets for usage instructions and integration.


## Reference

* [MediaPipe](https://mediapipe.dev/)

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)

# Translation and other improvements
Nikita Kiselov(https://github.com/kinivi)

---------------------------------------------------------
