import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Load test data
data_dict = pickle.load(open('./data.pickle', 'rb'))
x_test = np.asarray(data_dict['data'])
y_test = np.asarray(data_dict['labels'])

# Evaluate model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
conf_matrix = confusion_matrix(y_test, y_predict)

# Print results
print('Performance Metrics:')
print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1-score: {:.2f}'.format(f1))

labels_dict = {0: 'Blind', 1: 'Hello', 2: 'I love You', 3: 'Yes', 4:'No', 5:'thank you', 6:'Please',7:'Phone',8:'Horse',9:'ThumbsUp'
               ,10:'PeaceSign',11:'Walk', 12:'Up',13:'Four',14:'Sit',15:'Nine',16:'W',17:'X',18:'R',19:'Down',
               20:'Area', 21:'O',22:'Talk',23:'Hear',24:'Eat',25:'Male',26:'ThumbsDown',27:'Fish',28:'Lion',29:'King',30:'strong',31:'Sun',32:'Fireman'
               ,33:'Sky',34:'Gun',35:'Hospital',36:'Bed',37:'Snake',38:'Frog',39:'Light',40:'Pig',41:'Child',42:'Insect',43:'Black',44:'Red'
               ,45:'Juice',46:'Secret',47:'Stay',48:'Forget',49:'Police'
               }

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels_dict.values(), yticklabels=labels_dict.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Save evaluation results
with open('performance_analysis.txt', 'w') as f:
    f.write('Performance Metrics:\n')
    f.write('Accuracy: {:.2f}%\n'.format(accuracy * 100))
    f.write('Precision: {:.2f}\n'.format(precision))
    f.write('Recall: {:.2f}\n'.format(recall))
    f.write('F1-score: {:.2f}\n'.format(f1))
