import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data#Параметры
#print(X)
y = iris.target#[00001111222]
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))#считает отклонения оценщика


model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


y_train_2 = np.zeros((y_train.size, y_train.max()+1))#нулевая матрица
y_train_2[np.arange(y_train.size),y_train] = 1 # возвращает объект типа ndarray
model.fit(X_train, y_train_2, epochs=100, batch_size=10)#обучение модели
model.save('my_modell.h5')
print(y_train_2)

y_test_2 = np.zeros((y_test.size, y_test.max()+1))
y_test_2[np.arange(y_test.size),y_test] = 1
loss, accuracy = model.evaluate(X_test, y_test_2)

print('Accuracy:', accuracy)#визуализация данных

#model.save('my_model.h5')

input_data = []
for i in range(4):
    val = float(input(f"Введите значение {i+1}-го параметра: "))
    input_data.append(val)
input_data = np.array(input_data).reshape(1, -1)
prediction = model.predict(input_data)#генератор предсказаний
class_names = iris.target_names
predicted_class = class_names[np.argmax(prediction)]

print("Предсказанный класс: ", predicted_class)

colors = ['red', 'green', 'blue']
markers = ['o', 's', 'D']

for i in range(len(colors)):
    x = X[y == i][:, 0]
    y_ = X[y == i][:, 1]
    plt.scatter(x, y_, c=colors[i], marker=markers[i], label=iris.target_names[i])

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.show()