import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

print("started training model")
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.array([np.array(d) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

print("halfway 1")

_max_x_train=0
for i in x_train:
    if(_max_x_train<len(i)):
        _max_x_train=len(i)
print('_max_x_train')
print(_max_x_train)

temp_x_train=[]
for i in x_train:
    i=np.pad(i,(0,_max_x_train-len(i)),'constant',constant_values=(0))
    temp_x_train.append(i)

print('temp_x_train')
print(temp_x_train)

print("halfway 2")

model.fit(temp_x_train, y_train)

print("halfway 3")

_max_x_test=0
for i in x_test:
    if(_max_x_test<len(i)):
        _max_x_test=len(i)

print("_max_x_test")
print(_max_x_test)
f1=open("max_x_test.txt","w")
f1.write(str(_max_x_test))
f1.close()

print("halfway 4")

temp_x_test=[]
for i in x_test:
    i=np.pad(i,(0,_max_x_test-len(i)),'constant',constant_values=(0))
    temp_x_test.append(i)

print('temp_x_test')
print(temp_x_test)

print("halfway 5")

y_predict = model.predict(temp_x_test)

print("passed y_predict")

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
print("finished training model")
