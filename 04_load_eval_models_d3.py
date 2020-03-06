#################################################################################################################################
#Image Segmentation with "VOC 2012" Dataset######################################################################################
#################################################################################################################################



# Installation benötiger Packeges und Toolboxen----------------------------------------------------------------------------------
import os
import glob
import zipfile
import functools

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
import imageio
import scipy

from time import time
import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard
#--------------------------------------------------------------------------------------------------------------------------------









# Testausgabe zum Debuggen--------------------------------------------------------------------------------------------------------
print('\nInstallation packages erfolgleich!')
#---------------------------------------------------------------------------------------------------------------------------------









# Zuweisung von Ordnerpfaden-----------------------------------------------------------------------------------------------------
# Gewünschte Competition festlegen
competition_name = 'VOC2012'


# Festlegen von Pfaden
img_dir = os.path.join(competition_name, "JPEGImages")					# Pfad der jpg Bilder
mask_dir = os.path.join(competition_name, "SegmentationClass")				# Pfad der png Masken

train_txt_dir = os.path.join(competition_name, "Segmentation/train.txt")		# Pfad der txt-Datei mit Trainingsnamen
val_txt_dir = os.path.join(competition_name, "Segmentation/val.txt")			# Pfad der txt-Datei mit Validierungsnamen
#trainval_txt_dir = os.path.join(competition_name, "Segmentation/trainval.txt")		# Pfad der txt-Datei mit Trainings- und Validierungsnamen

np_image_dir = os.path.join(competition_name, "np_Images")				# Pfad aller Bilder im Ordner np_Images
np_SegmentationClass_dir = os.path.join(competition_name, "np_SegmentationClass")	# Pfad aller Masken im Ordner np_SegmentationClass

train_pic_path = os.path.join(competition_name, "np_train_Images")			# Pfad Trainingsbilder
train_mask_path = os.path.join(competition_name, "np_train_SegmentationClass")		# Pfad Trainingsmasken
val_pic_path = os.path.join(competition_name, "np_val_Images")				# Pfad Validierunsbilder
val_mask_path = os.path.join(competition_name, "np_val_SegmentationClass")		# Pfad fValidierungsmasken


np_val_mask_save_dir = os.path.join(competition_name, "np_val_mask_array")
np_val_pic_save_dir = os.path.join(competition_name, "np_val_pic_array")

# Ausgabe der Festgelegten Pfade
#print(train_pic_path)
#print(train_mask_path)
#print(val_pic_path)
#print(val_mask_path)
#----------------------------------------------------------------------------------------------------------------------------------









# Festlegung weiterer Parameter-----------------------------------------------------------------------------------------------------
shuffle = False
img_size = (192, 192, 3)
batch_size = 5						
#epochs = 5



'''
# Erstellen des Trainingsnamen und Ordners
note = 'standard_tf'
architecture = 'U_Net'
model_name = '{}_voc_2012_Seg_{}_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'), architecture, note)
os.makedirs('/home/lukas/Dokumente/Trainings/{}'.format(model_name))

#-----------------------------------------------------------------------------------------------------------------------------------
'''



# Klasse Daten-------------------------------------------------------------------------------------------------------------
class data_generator(tf.keras.utils.Sequence):


    def __init__(self, pic_path, mask_path, batch_size, shuffle):
        
        self.pic_path = pic_path							# Initialisierung
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.shuffle = shuffle


        pic_list = []
        mask_list = []

        for root, _, files in os.walk(pic_path):					# Einlesen aller Bildernamen
            for filename in files:
            	pic_list.append(os.path.join(root, filename))



        for root, _, files in os.walk(mask_path):					# Einlesen aller Maskennamen
            for filename in files:	
                mask_list.append(os.path.join(root, filename))


        if self.shuffle == True:
          tmp = list(zip(pic_list, mask_list))						# Shufflen der Listen
          random.shuffle(tmp)
          pic_list, mask_list = zip(*tmp)
        #print("pic_list[0]:", pic_list[0])
        #print("mask_list[0]:", mask_list[0])
        #print("Anzahl Trainingsbilder:", len(pic_list))
        #print("Anzahl Trainingsmasken:", len(mask_list))
        self.pic_list =pic_list#[5]							# Init geshuffelte Bilderliste
        self.mask_list = mask_list#[5]							# Init geshuffelte Maskenliste
        #print("pic_list[0]:", pic_list[0])
        #print("mask_list[0]:", mask_list[0])
        #print("Anzahl Validierungsbilder:", len(pic_list))
        #print("Anzahl Validierungsmasken:", len(mask_list))


    def __len__(self):
        return int(np.ceil(len(self.pic_list) / self.batch_size))			# Anzahl Durchläufe pro Epoche




    def __getitem__(self, idx):								# Laden x_batch (Bilder), y_batch (Masken)
        #print("idx:", idx)
        #print("batch_size:", batch_size)
        X_batch = np.load( self.pic_list[idx*self.batch_size] )[np.newaxis, ...]	
        y_batch = np.load( self.mask_list[idx*self.batch_size] )[np.newaxis, ...]

        for i in range(idx*self.batch_size+1, (idx+1)*self.batch_size):
            try:
                pic = np.load(self.pic_list[i]) 
                mask = np.load(self.mask_list[i])
            except IndexError:
                break
            X_batch = np.append(X_batch, pic[np.newaxis, ...], axis=0)			# Shape (batchsize, höhe, breite, 3), Werte [0...1], 3 -> RGB
            y_batch = np.append(y_batch, mask[np.newaxis, ...], axis=0)			# Shape (batchsize, höhe, breite, 21), Werte [0, 1], 21 -> Anz. Klassen
        
        return (X_batch, y_batch)





# Testinstanzierung Klasse data_generator
# Falls einkommentiert, werden alle Masken und bilder eingelesen
#test_gen = data_generator(val_pic_path, val_mask_path, batch_size, shuffle)
#
#x_val, y_val = test_gen.__getitem__(0)
#print('x_batch shape:', x_val.shape)
#print('y_batch shape:', y_val.shape)
##print('x_batch', x_val)
##print('y_batch', y_val)
#
# Speichern der erstellten Arrays
#np.save(np_val_pic_save_dir, x_val)		# Validierungsbilder
#np.save(np_val_mask_save_dir, y_val)		# Validierungsmasken

# Laden der erstellten Arrays
np_val_mask_save_dir = os.path.join(competition_name, "np_val_mask_array.npy")
np_val_pic_save_dir = os.path.join(competition_name, "np_val_pic_array.npy")



x_val = np.load(np_val_pic_save_dir)
y_val = np.load(np_val_mask_save_dir)


print("\nx_val_shape:", x_val.shape)
print('y_val_shape:', y_val.shape)
# -------------------------------------------------------------------------------------------------------------------------------

'''





#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################




# Definieren der Blockstrucktur---------------------------------------------------------------------------------------------------

def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder




# Definieren der Modellstrucktur------------------------------------------------------------------------------------------------------------
with tf.name_scope('Inputlayer'):
  inputs = layers.Input(shape=img_size)
  # 256


with tf.name_scope('Encoder_1'):
  encoder0_pool, encoder0 = encoder_block(inputs, 32) # 32
  # 128


with tf.name_scope('Encoder_2'):
  encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
  # 64


with tf.name_scope('Encoder_3'):
  encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 128
  # 32


with tf.name_scope('Encoder_4'):
  encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 256
  # 16


with tf.name_scope('Encoder5'):
  encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 512
  # 8


with tf.name_scope('Center'):
  center = conv_block(encoder4_pool, 1024) # 1024
  # center


with tf.name_scope('Decoder_1'):
  decoder4 = decoder_block(center, encoder4, 512) # 512 
  # 16


with tf.name_scope('Decoder_2'):
  decoder3 = decoder_block(decoder4, encoder3, 256) # 256
  # 32


with tf.name_scope('Decoder_3'):
  decoder2 = decoder_block(decoder3, encoder2, 128) # 128
  # 64


with tf.name_scope('Decoder_4'):
  decoder1 = decoder_block(decoder2, encoder1, 64) # 64
  # 128


with tf.name_scope('Decoder_5'):
  decoder0 = decoder_block(decoder1, encoder0, 32) # 32
  # 256


with tf.name_scope('Outputlayer'):
  outputs = layers.Conv2D(21, (1, 1), activation='softmax')(decoder0)
  #256




# Definieren des Modells-----------------------------------------------------------------------------------------------------------------
model = models.Model(inputs=[inputs], outputs=[outputs])

'''
def weighted_crossentropy(y_true, y_pred):
    weights = np.array([0.05, 67/90, 67/79, 67/103, 67/72, 67/96, 67/74, 67/127, 67/119, 67/123, 67/71, 67/75, 67/128, 67/79, 67/76, 67/446, 67/85, 67/57, 67/90, 67/84, 67/74])
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss



# Definiton dice-loss und dice-Koeffizient für die Gewichtung der Pixel (bezogen auf for/background)----------------------------------------
def dice_coeff(y_true, y_pred):
    smooth = 1.
    print("Format y_pred:",y_pred.shape)
    print("Format y_true:",y_true.shape)
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# Definieren der Kostenfunktion (Kreuzentropie + dice_loss)
def bce_dice_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) #+ dice_loss(y_true, y_pred)
    return loss

#-------------------------------------------------------------------------------------------------------------------------------------------

'''
# Instanzieren des Modells
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])




# Erstellung Tensorboard-Graph und Modellspeicherung--------------------------------------------------------------------------------------
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/home/lukas/Dokumente/Trainings/{}/logs'.format(model_name))

model.summary()

save_model_path = '/home/lukas/Dokumente/Trainings/{}/weights.hdf5'.format(model_name) 

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)



# Initialisierung der Generatoren--------------------------------------------------------------------------------------------------------
training_generator   = data_generator(train_pic_path, train_mask_path, batch_size)
'''

'''



# Durchführung des Trainings-------------------------------------------------------------------------------------------------------------
history = model.fit_generator(
                   training_generator,
                   epochs=epochs,
                   validation_data=validation_generator,
                   callbacks=[tensorboard, cp],
                   max_queue_size=128,
                   use_multiprocessing=True)






# Ausgabe des Losses über den Epochen (Wert der Kostenfunktion)--------------------------------------------------------------------------
dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()




'''

# Farbpallette für den VOC-2012 Datensatz------------------------------------------------------------------------------------------------
color_dict = {0:[0, 0, 0], 1:[128, 0, 0], 2:[0, 128, 0], 3:[128, 128, 0], 4:[0, 0, 128], 5:[128, 0, 128],
                      6:[0, 128, 128], 7:[128, 128, 128], 8:[64, 0, 0], 9:[192, 0, 0], 10:[64, 128, 0],
                      11:[192, 128, 0], 12:[64, 0, 128], 13:[192, 0, 128], 14:[64, 128, 128], 15:[192, 128, 128],
                      16:[0, 64, 0], 17:[128, 64, 0], 18:[0, 192, 0], 19:[128, 192, 0], 20:[0, 64, 128]}



# Farbpallette für die one-hot Codierung---------------------------------------------------------------------------------------------------
color_dict_neo = {0:[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  1:[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  2:[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  3:[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  4:[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  5:[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  6:[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  7:[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  8:[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  9:[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 10:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 11:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 12:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 13:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 14:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 15:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 16:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 17:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 18:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                 19:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                 20:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}



# Liste aller Klassen im VOC-2012-Datensatz------------------------------------------------------------------------------------------------
class_dict = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "char", "cow", "dinigtabe", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor")


class_freq = np.array([1442, 90, 79, 103, 72, 96, 74, 127, 119, 123, 71, 75, 128, 79, 76, 446, 85, 57, 90, 84, 74])
# background 14449, person 445
print("class_frequz_shape", class_freq.shape)
# Konvertierung zu rgb-Format----------------------------------------------------------------------------------------------------------------
def encoded_to_np_rgb(onehot, colormap = color_dict):
    # Funktion zum encoden von Trainingsmasken 
    # (h, w, 21), [Null/Eins oder Prob] je Klasse  ->  (h, w, 3), RGB-Wert je Pixel
 
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)











# Pfaddefinitionen und Abfrage nach zu ladendem Model---------------------------------------------------------------------------------------
print("\nBilder Pfad:",val_pic_path)
print("Masken Pfad:",val_mask_path)
validation_generator = data_generator(val_pic_path, val_mask_path, batch_size, shuffle)

#print("validation_generator_type", type(validation_generator))
#print("validation_generator", validation_generator)

PATH=input('\nBitte geben sie den Ordnernamen des zu ladendne Modells ein:')
train_path = '/home/lukas/Dokumente/Trainings/'
save_model_path = []
save_model_path.append(os.path.join(train_path, "{}/weights.hdf5".format(PATH)))
print('Zu ladendes Modell:', save_model_path)

val_pic_list = []
val_mask_list = []

for root, _, files in os.walk(val_pic_path):					# Einlesen aller Validierungs-Bildernamen 
    for filename in files:
        val_pic_list.append(os.path.join(root, filename))

for root, _, files in os.walk(val_mask_path):					# Einlesen aller Validierungs-Maskennamen 
    for filename in files:	
        val_mask_list.append(os.path.join(root, filename))

masken = np.load(val_mask_list[0])[np.newaxis, ...] 				# Laden der ersten Validierungsmaske (als Dummy)
print("maske", masken.shape)							# Ausgabe Validierungsmasken-Shpae
print("val_mask_anz", len(val_mask_list))					# Ausgabe Anzahl Validierungsmasken


print('\n')














# Laden des Modells-------------------------------------------------------------------------------------------------------------------------
model = models.load_model(save_model_path[0], custom_objects={'bce_dice_loss': bce_dice_loss,
                                                           'dice_loss': dice_loss,
                                                            'weighted_crossentropy':weighted_crossentropy})
       

 
# Vorhersagen mit dem geladenen Modell berechnen--------------------------------------------------------------------------------------------
vorhersagen = model.predict_generator(
                   validation_generator,
                   max_queue_size=10,
                   workers=12,
                   use_multiprocessing=True)


# Ausgabe der wichtigsten Vorhersage-Eigenschaften
print('\nVorhersagen_shape', vorhersagen.shape)
print("vorhersagen_type", type(vorhersagen))
print('Vorhersagen sind nach softmax Prob:\n', vorhersagen[0:2, 0:2, 0:2, 0:2])

# In "vorhersagen" stehen jetzt alle (!) prognostizierten Masken für den Validierungsdatensatz

# Umwandlung der Vorhersagen von Prob nach One-Hot
max_value = np.argmax(vorhersagen, axis= -1)					# Auslesen der höchsten Wahrscheinlichkeiten je Bildpunkt je Bild
print('\nmax_value_shape', max_value.shape)					# Ausgabe Shape von max_value


# In max_value steht für alle prog. Masken an jedem Pixelwert die Klasse [0:20], die dem Pixelwert (mit der höchsten (prog.) Wahrscheinlichkeit) zugeordnet werden soll



















## Testbereich

def iou_t(y_true, y_pred):
  # Funktion zur Berechnung iou je Klasse (mit Background)
  num_classes = (y_pred.shape[2])					# Festlegung Anzahl Klassen
  #print('num_classes', num_classes)
  iou = np.zeros(num_classes)			
  for i in range(num_classes):
      y_true_n = y_true[:,:,i]						# Auswahl Klasse i
      y_pred_n = y_pred[:,:,i]						# Auswahl Klasse i
      
      y_true_n_f = np.reshape(y_true_n, [-1])   			# Umwandlung in Vektor
      y_pred_n_f = np.reshape(y_pred_n, [-1])  				# Umwandlung in Vektor
      
      if np.sum(y_true_n_f) !=0:
        intersection = np.sum(y_true_n_f * y_pred_n_f)			# Berechnung Überlappung
        union = np.sum(y_true_n_f) + np.sum(y_pred_n_f) - intersection	# Berechnung Union
        iou[i] = intersection/union
      else:
        iou[i]=0
  return iou









one_hot_pred = np.zeros(max_value.shape[1:3]+(21,) ,dtype=np.int)		# Anlegen eines (!) One-Hot Bildes (init mit zeros)
print('\none_hot_pred_shape', one_hot_pred.shape)				# Ausgabe shape: (1,192,192,21)

iou = np.zeros(21)

for i in range(len(val_mask_list)):
  y_true = y_val[i,:,:,:]
  m=0
  for m in range(img_size[0]):							# Umwandlung eines Bilds von Prob -> One-Hot
    n=0
    for n in range(img_size[0]):
      k=0
      for k in color_dict_neo.keys():
        if max_value[i,m,n]==k:
          one_hot_pred[m,n,:] = color_dict_neo[k]
  
  # y_true (192,192,21), one-hot-encoded
  # y_pred = one_hod_pred (192,192,21), one-hot-encoded
  iou_n = np.zeros(21)
  iou_n = iou_t(y_true, one_hot_pred)
#  iou_n = iou_t(y_true, y_true)
  iou = iou + iou_n
  
  if (i%20)==0:
    print("iou für Bild_{}".format(i)," berechnet")


iou=iou/class_freq

print('\n Intersection over union:')
for i in range(21):
  print(class_dict[i], iou[i])


print("\nmIoU:", np.mean(iou))






	
## Testbereich Ende





print("\nDie ganze Gaudi für ein Bild:")

output = np.zeros(max_value.shape[1:3]+(21,) ,dtype=np.int)			# Anlegen eines (!) One-Hot Bildes (init mit zeros)
print('\noutput', output.shape)							# Ausgabe der Shape des One-Hot Bildes


for m in range(img_size[0]):							# Umwandlung eines Bilds von Prob -> One-Hot
  n=0
  for n in range(img_size[0]):
    k=0
    for k in color_dict_neo.keys():
      if max_value[0,m,n]==k:
        output[m,n,:] = color_dict_neo[k]



print('n',n)

print('output_one_hot', output.shape)

print('y_ture:', type(vorhersagen))
print('y_pred:', type(output))



outputs = output [np.newaxis, ...] 
print("output_s:", outputs.shape)



'''
for i in range (test.shape[0]):
  m=0
  for m in range(img_size[0]):
    n=0
    for n in range(img_size[0]):
      k=0
      for k in color_dict_neo.keys():
        if test[i,m,n]==k:
          output[m,n,:] = color_dict_neo[k]#[np.newaxis, ...]
          if i==0:
            print('output_shape', output.shape)
          output_s = output[np.newaxis, ...]
          if i==0:
            print('output_s_shape', output_s.shape)
          outputs = np.append(outputs, output_s, axis=0)

outputs = outputs[1:,:,:,:]

print("outputs", outputs.shape)
'''


















for i in range(len(val_mask_list)-1440):
    maske = np.load(val_mask_list[i])[np.newaxis, ...]
    #print("maske_n", maske.shape)
    masken = np.append(masken, maske, axis=0)

masken = masken[1:len(val_mask_list)-1440,:,:,:] 				# Von 1:len(val_mask_list), weil das 0. Element ein Dummy für die Shape Ausgabe war

print("masken[0]_shape", masken[0].shape)












def iou(y_true, y_pred):
  # Funktion zur Berechnung iou je Klasse (mit Background)
  num_classes = (y_pred.shape[2])					# Festlegung Anzahl Klassen
  print('num_classes', num_classes)
  iou = np.zeros(num_classes)			
  for i in range(num_classes):
      y_true_n = y_true[:,:,i]						# Auswahl Klasse i
      y_pred_n = y_pred[:,:,i]						# Auswahl Klasse i
      
      y_true_n_f = np.reshape(y_true_n, [-1])   			# Umwandlung in Vektor
      y_pred_n_f = np.reshape(y_pred_n, [-1])  				# Umwandlung in Vektor
      
      if np.sum(y_true_n_f) !=0:
        intersection = np.sum(y_true_n_f * y_pred_n_f)			# Berechnung Überlappung
        union = np.sum(y_true_n_f) + np.sum(y_pred_n_f) - intersection	# Berechnung Union
        iou[i] = intersection/union
      else:
        iou[i]=2
  return iou

iou = iou(masken[0], output)

print('Intersection over union:')
for i in range(21):
  print(class_dict[i], iou[i])




def iou_2(y_true, y_pred):
  # Funktion zur Berechnung iou je Klasse (ohne Background)
  num_classes = (y_pred.shape[2])		# Festlegung Anzahl Klassen
  print('num_classes', num_classes)
  iou = np.zeros(num_classes)			
  for i in range(1, num_classes):
      y_true_n = y_true[:,:,:,i]					# Auswahl Klasse i
      y_pred_n = y_pred[:,:,:,i]					# Auswahl Klasse i
      
      y_true_n_f = np.reshape(y_true_n, [-1])   			# Umwandlung in Vektor
      y_pred_n_f = np.reshape(y_pred_n, [-1])  				# Umwandlung in Vektor
      
      if np.sum(y_true_n_f) !=0:
        intersection = np.sum(y_true_n_f * y_pred_n_f)			# Berechnung Überlappung
        union = np.sum(y_true_n_f) + np.sum(y_pred_n_f) - intersection	# Berechnung Union
        iou[i] = intersection/union
      else:
        iou[i]=22222222222
  return iou

'''iou = iou_2(masken[:3], output)

print('Intersection over union:')
for i in range(21):
  print(class_dict[i], iou[i])
'''




print('Val_pic_anz:', len(val_pic_list))
print('Val_mask_anz:', len(val_mask_list))
print('Val_pic_bsp:', val_pic_list[0])
print('Val_mask_bsp:', val_mask_list[0])

print('Masken', masken.shape)

print('Masken', type(masken))


# Ausgabe einer Bilder mit dazugehörigen (Ground-Truth) und errechneten Masken------------------------------------------------------------------
plt.figure(figsize=(10,20))
for i in range(4):
  
  pic = np.load(val_pic_list[i])
  plt.subplot(4, 3, 3 * i + 1)
  plt.imshow(pic)
  if i==0:
    plt.title("Val-Bilder")


  mask = masken[i]
  encoded_mask= mask	
  decoded_mask = encoded_to_np_rgb(encoded_mask)

  plt.subplot(4, 3, 3 * i + 2)
  plt.imshow(decoded_mask)
  if i==0:
    plt.title("Val-Masken")


  encoded_pred = vorhersagen[i,:,:,:]
  if i==0:	
    encoded_pred = output
  decoded_pred = encoded_to_np_rgb(encoded_pred)
  
  plt.subplot(4, 3, 3 * i + 3)
  plt.imshow(decoded_pred)
  if i==0:
    plt.title("Vorhersagen")

plt.suptitle("Validierungs - Bilder,Masken und vorhergesagte Masken")
plt.show()



'''
def iou(y_true, y_pred):
  # Funktion zur Berechnung iou je Klasse (ohne Background)
  num_classes = (y_pred.shape[2])		# Festlegung Anzahl Klassen
  print('num_classes', num_classes)
  iou = np.zeros(num_classes)			
  for i in range(1, num_classes):
      y_true_n = y_true[:,:,:,i]					# Auswahl Klasse i
      y_pred_n = y_pred[:,:,:,i]					# Auswahl Klasse i
      
      y_true_n_f = np.reshape(y_true_n, [-1])   			# Umwandlung in Vektor
      y_pred_n_f = np.reshape(y_pred_n, [-1])  				# Umwandlung in Vektor
      
      if np.sum(y_true_n_f) !=0:
        intersection = np.sum(y_true_n_f * y_pred_n_f)			# Berechnung Überlappung
        union = np.sum(y_true_n_f) + np.sum(y_pred_n_f) - intersection	# Berechnung Union
        iou[i] = intersection/union
      else:
        iou[i]=22222222222
  return iou

iou = iou(masken[0], output)

print('Intersection over union:')
for i in range(21):
  print(class_dict[i], iou[i])






# Funktion noch nicht getestet!
def ev_miou(y_true, y_pred):
    # Funktion zum Berechnen des iou je Klasse, ohne Background	
    num_classes = (y_true.shape[3])
    iou = []
    m_iou = 0.0
    for i in range(1, num_classes):
      y_true_n = y_true[:,:,:,i]
      y_pred_n = y_pred[:,:,:,i]
      y_true_n_f = tf.reshape(y_true_n, [-1])
      y_pred_n_f = tf.reshape(y_pred_n, [-1])
       
      intersection = tf.reduce_sum(y_true_n_f * y_pred_n_f)
      union = tf.reduce_sum(y_true_n_f) + tf.reduce_sum(y_pred_n_f) - intersection
      iou_n = intersection/union   
      #iou = np.append(iou_n)
      m_iou = m_iou + iou_n
    m_iou = m_iou/num_classes      
      
    return m_iou, iou



m_iou, iou = ev_miou(masken, vorhersagen)

print("m_iou", m_iou)
print("iou", iou)



def Mean_IOU_tensorflow_1(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)





# Ausgabe einer Bilder mit dazugehörigen (Ground-Truth) und errechneten Masken
plt.figure(figsize=(10, 20))
for i in range(4):
#  batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
#  img = batch_of_imgs[0]
#  predicted_label = model.predict(batch_of_imgs)[0]

  plt.subplot(4, 2, 2 * i + 1)
  plt.imshow(img)
  if i==0:
    plt.title("Testbilder")

  #plt.subplot(5, 3, 3 * i + 2)
  #plt.imshow(label[0, :, :, 0])
  #plt.title("Actual Mask")
  plt.subplot(4, 2, 2 * i + 2)
  plt.imshow(predicted_label[:, :, 0])
  if i==0:
    plt.title("Segmentierungen")

plt.suptitle("Testbilder und vorhergesagte Masken")
plt.show()














temp_ds2 = get_baseline_dataset(x_test_filenames[:5], 
                                x_test_filenames[:5], #(eigentlich labels, gibt es aber für Testbilder nicht!)
                                preproc_fn=test_preprocessing_fn,
                                batch_size=5)

# Erstellen eines Ausgabedatensatzes 
data_aug_iter2 = temp_ds2.make_one_shot_iterator()
next_element2 = data_aug_iter2.get_next()

# Ausgabe einer Bilder mit dazugehörigen (Ground-Truth) und errechneten Masken
plt.figure(figsize=(10, 20))
for i in range(4):
  batch_of_imgs, label = tf.keras.backend.get_session().run(next_element2)
  img = batch_of_imgs[0]
  predicted_label = model.predict(batch_of_imgs)[0]

  plt.subplot(4, 2, 2 * i + 1)
  plt.imshow(img)
  if i==0:
    plt.title("Testbilder")

  #plt.subplot(5, 3, 3 * i + 2)
  #plt.imshow(label[0, :, :, 0])
  #plt.title("Actual Mask")
  plt.subplot(4, 2, 2 * i + 2)
  plt.imshow(predicted_label[:, :, 0])
  if i==0:
    plt.title("Segmentierungen")

plt.suptitle("Testbilder und vorhergesagte Masken")
plt.show()

















batch_img,batch_mask = next(testing_gen)
pred_all= model.predict(batch_img)
np.shape(pred_all)





for i in range(0,np.shape(pred_all)[0]):
    
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(batch_img[i])
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)
    
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(onehot_to_rgb(batch_mask[i],id2code))
    ax2.grid(b=None)
    
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Predicted labels')
    ax3.imshow(onehot_to_rgb(pred_all[i],id2code))
    ax3.grid(b=None)
    
    plt.show()













# Klassen des Datensatzes

(1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor)

class_dict = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "char", "cow", "dinigtabe", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor")






# Default Farbpallette für den VOC-2012 Datensatz-------------------------------------------------------------------------------
color_dict = {0:[0, 0, 0], 1:[128, 0, 0], 2:[0, 128, 0], 3:[128, 128, 0], 4:[0, 0, 128], 5:[128, 0, 128],
                      6:[0, 128, 128], 7:[128, 128, 128], 8:[64, 0, 0], 9:[192, 0, 0], 10:[64, 128, 0],
                      11:[192, 128, 0], 12:[64, 0, 128], 13:[192, 0, 128], 14:[64, 128, 128], 15:[192, 128, 128],
                      16:[0, 64, 0], 17:[128, 64, 0], 18:[0, 192, 0], 19:[128, 192, 0], 20:[0, 64, 128]}









def encoded_to_np_rgb(onehot, colormap = color_dict):
    # Funktion zum encoden von Trainingsmasken 
    # (h, w, 21), Null/Eins je Klasse 	   ->      (h, w, 3), RGB-Wert je Pixel
 
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)






'''










