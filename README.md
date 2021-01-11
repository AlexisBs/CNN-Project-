# CNN-Project-
<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/logos.PNG" style="max-width:100%;"></p>
<p align="center">
<h1>DATA MINING</h1>
<p align="center">
<br><strong>Tecnológico Nacional de México</strong>
<br><strong>Instituto Tecnológico de Tijuana</strong>
<br><strong>Subdirección académica</strong>
<br><strong>Departamento de Sistemas y Computación</strong>
<br><strong>Semestre: Agosto - Diciembre 2020</strong>
<br><strong>Ingeniería en Tecnologías de la Información y Comunicaciones</strong>
<br><strong>Materia: DATA MINING </strong>
<br><strong>Barraza Sierra Alexis Fernando 16212532</strong>
<br><strong>Docente: Dr. Jose Christian Romero Hernandez</strong>
</p>

<h1>Index</h1>
<ol>
<li><a href = "#Practica1" target="_self"> Neural Networks </a>
<li><a href = "#Practica2" target="_self"> Learning methods </a>
<li><a href = "#Practica3" target="_self"> Deep Learning </a>
<li><a href = "#Practica4" target="_self"> Convolutional Neural Network </a>
<li><a href = "#Practica5" target="_self"> Architecture </a>
<li><a href = "#Practica6" target="_self"> Software to Use </a>
<li><a href = "#Practica7" target="_self"> Project </a>

</ol>

<h1>Neural Networks</h1>
<a name = "Practica1"> <h3>Neural Networks</h3> </a>
<p align="justify">Neural networks are computational systems, inspired by the neurons that make up the brain of animals, providing computers with artificial intelligence. They are made up of basic units called neurons that connect to each other forming the neural network.
Artificial neural networks can be described as models with at least two layers: one input and one output, as well as, in most cases, other intermediate layers (hidden layers). The more complex the problem that the neural network must help solve, the more layers are required. In each of them you can find numerous specialized artificial neurons.
</p>
<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/RedNeuronal.png" style="max-width:100%;"></p>
<p align="center">

<a name = "Practica2"><h1>Learning methods</h1>  </a>
<p align="justify">In order for the connections in artificial neural networks to be properly linked and to be able to solve the proposed tasks, they must be previously trained. For this, there are two basic processes:
</p>
<h2>Supervised learning</h2>
<p align="justify">In this type of learning, a specific result is defined for different inputs or input content. If, for example, the system is expected to recognize photos of cats, there will be people who check the classification made by the system and let you know which images you have hit and which you have not. In this way the weights in the network can be modified and the algorithm optimized.</p>
<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/aprendizaje_supervisado.png" style="max-width:100%;"></p>
<p align="center">
  
<h2>Unsupervised learning</h2>
<p align="justify">In unsupervised learning, the result of the task is not given, but rather the system finds out exclusively from the input information. Hebb's law or adaptive resonance theory intervenes in this process.</p>
<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/aprendizaje_no_supervisado.png" style="max-width:100%;"></p>
<p align="center">

<a name = "Practica3"><h1>Deep Learning</h1>  </a>
<p align="justify">Deep Learning carries out the Machine Learning process using an artificial neural network that is made up of a number of hierarchical levels. At the initial level of the hierarchy the network learns something simple and then sends this information to the next level. The next level takes this simple information, combines it, composes some slightly more complex information, and passes it to the third level, and so on.</p>
<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/deep.jpg" style="max-width:100%;"></p>
<p align="center">
  
Below are some of the main “real” problems that different companies are applying Deep Learning to today:
* Using images instead of keywords to search for a company's products, or similar items.
* Identify company brands and logos in photos posted on social media.
* Real-time monitoring of reactions in online channels during product launches.
* Ad targeting and prediction of customer preferences.
* Identification of potential clients.
* Fraud detection, customer recommendations, customer relationship management, etc.
* Better understanding of diseases, disease mutations, and gene therapies.
* Analysis of medical images, such as X-rays and MRIs, increasing diagnostic precision, in less time and at a lower cost than traditional methods.
* Detection, prediction and prevention of sophisticated threats in real time in the field of cybersecurity.
* Speech recognition.
* Video classification.

<a name = "Practica4"><h1>Convolutional Neural Network</h1>  </a>
<p align="justify">CNN is a type of Artificial Neural Network with supervised learning that processes its layers imitating the visual cortex of the human eye to identify different characteristics in the inputs that ultimately make it able to identify objects and "see". For this, CNN contains several specialized hidden layers and with a hierarchy: this means that the first layers can detect lines, curves and are specialized until they reach deeper layers that recognize complex shapes such as a face or the silhouette of an animal .</p>
<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/cnns-01_1.png" style="max-width:100%;"></p>
<p align="center">
 
 <a name = "Practica5"><h1> Architecture</h1>  </a>

<p align="center"><img src="https://github.com/OsziiRk/Recursos_Bigdata/blob/master/1_vkQ0hXDaQv57sALXAJquxA.jpeg" style="max-width:100%;"></p>
<p align="center">
  
<h3>Entry</h3> 
<p align="justify"> It will be the pixels of the image. They will be height, width and depth will be 1 only color or 3 for Red, Green, Blue. </p>
<h3> Convolution Layer</h3> 
<p align = "justify"> processes the output of neurons that are connected in input "local regions" (that is, nearby pixels), calculating the dot product between their weights (pixel value) and a small region to which they are connected to the input volume. Here we will use for example 32 filters or the amount that we decide and that will be the output volume. </p>
<h3> Relu Cape </h3>
<p align = "justify"> will apply the activation function on the elements of the array. </p>
<h3> Pool or Subsampling </h3>
<p align = "justify"> Will make a reduction in the height and width dimensions, but the depth is maintained. </p>
<h3> Traditional Cape </h3>
<p align = "justify"> feedforward neuron network that will connect with the last subsampling layer and end with the number of neurons that we want to classify. </p>

 <a name = "Practica6"><h1> Software to Use</h1>  </a>
<h3>TensorFlow</h3> 
<p align="justify"> TensorFlow is an open source software library for numerical computing, using data flow graphs. The nodes in the graphs represent mathematical operations, while the edges of the graphs represent the multidimensional data matrices (tensors) communicated between them.
TensorFlow is a great platform for building and training neural networks, which allow detecting and deciphering patterns and correlations, analogous to the learning and reasoning used by humans.</p>
<h3>Keras</h3> 
<p align="justify"> Keras is a library that works at the model level: it provides building blocks upon which complex deep learning models can be built. Unlike frameworks, this open source software is not used for simple low-level operations, but instead uses the linked machine learning framework libraries, which in a way act as a backend engine for Keras. The layers of the neural network to be configured are related to each other according to the modular principle, without the Keras user having to understand or directly control the backend of the chosen framework.
As we have mentioned, Keras relies especially on the TensorFlow, Theano and Microsoft Cognitive Toolkit tools, for which there are ready-to-use interfaces that allow quick and intuitive access to the corresponding backend.</p>

<a name = "Practica7"><h1> Project - Birds or planes?</h1>  </a>
<p align="justify"> Within the following project, a 2-layer neural network will be made using tensorflow and Keras, with a file with 10,000 images of birds and airplanes, a network will be made that learns and identifies all the differences between the images by means of each of the pixels that count the picture. With the program at the time of giving an image and already trained the neuron will have to deduce if it is a bird or an airplane.</p>

```python
import numpy as np ##importacion de librerias numpy para manejo de datos
import pandas as pd#importacion pandas para estructura y manejo de los datos

import matplotlib.pyplot as plt #importacion para manejo de datos e intrpretacion

from sklearn.model_selection import train_test_split #libreria de tensorflow sklearn para entrenamiento
from sklearn.metrics import classification_report, accuracy_score #libreria para la clasificacion de los datos


from keras.models import Sequential #ckeras para creacion de neurona
from keras.layers import Dense, Activation,Input,Dropout,Convolution2D,MaxPooling2D,Flatten# creacion de capas y pesos

%matplotlib inline

picture_size=32
channels = 'rgb'

input_columns= []
for color in channels:
    input_columns.extend(['%s%i'%(color,i)
                         for i in range(picture_size**2)])
                         
data.shape #informacion de datos y columnas filas

train, test = train_test_split(data, test_size=0.2) #Separacion de datos para test y entrenado, con un 20% de separacion
sets = (
    ('train', train),
    ('test', test), #guardo los datos en una lista
)

for set_name, set_data in sets:
    print('#' * 20, set_name, 'labels', '#' * 20)
    print(set_data.label.value_counts()) #cuantos valores entrenaste y cuentos no
    print()

def extract_inputs(dataset):
    return dataset[input_columns].values #conversion de las imagene a numeros para que entienda la red neuronal
model = Sequential([
   \

    Dense(50, input_shape=(len(input_columns), )),
    Activation('tanh'), #creacion de las neuronas con 50

    Dense(50),
    Activation('tanh'), #segunda capa con 50 

    
    Dense(1),
    Activation('sigmoid'), #salida entra 0 y 1
])

model.compile(
    optimizer='adam', #optimizador
    loss='binary_crossentropy', #calcular el error
    metrics=['accuracy',], #porcentaje que sea por porcentaje la salida
)

model.fit(
    extract_inputs(train),  #saca los datos del entrenamiento
    train.label.values, 
    epochs=5, #cuantas veces va a pasar po rla neurona
    batch_size=128, #cuantas imagenes ajustamos el peso
)  ## correr el entrenamiento, agarramos los datos previamente para entrenarlo, 5 veces el recorrido

for set_name, set_data in sets:
    labels = set_data.label.values
    predicted_labels = np.rint(model.predict(extract_inputs(set_data)))

    print('#' * 25, set_name, '#' * 25)
    print('accuracy', accuracy_score(labels, predicted_labels))
    print(classification_report(labels, predicted_labels))
    
test_with_predictions = test.copy()
test_with_predictions['prediction'] = model.predict(extract_inputs(test_with_predictions))
test_with_predictions['predicted_label'] = np.rint(test_with_predictions.prediction)
is_correct = test_with_predictions.label == test_with_predictions.predicted_label


show_images(test_with_predictions[is_correct].sample(5), title='prediction') #cuales me dio buena


show_images(test_with_predictions[~is_correct].sample(5), title='prediction')#cuales error
from PIL import Image

def classify_pictures(pictures_paths):
    raw_pictures_data = []
    
    for picture_path in pictures_paths:
        picture = Image.open(picture_path)

        picture = picture.resize((picture_size, picture_size), Image.ANTIALIAS)
        picture_data = np.array(list(zip(*picture.getdata()))).reshape(len(input_columns))
    
        raw_pictures_data.append(picture_data)
    
    pictures_data = pd.DataFrame(raw_pictures_data, columns=input_columns)

    pictures_data[input_columns] = pictures_data[input_columns].values / 255

    pictures_data['prediction'] = model.predict(extract_inputs(pictures_data))
    
    show_images(pictures_data, title='prediction') #creacion de una funcion que me convirte mis iamgenes en pixeles para proximo preddicion
    
classify_pictures([
    '/Users/alexi/Downloads/pajaro1.jpg'])
  
```

  
  


 
 



