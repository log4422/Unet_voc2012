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


# Ausgabe der Festgelegten Pfade
#print(train_pic_path)
#print(train_mask_path)
#print(val_pic_path)
#print(val_mask_path)
#----------------------------------------------------------------------------------------------------------------------------------









# Festlegung weiterer Parameter-----------------------------------------------------------------------------------------------------
img_size = (192, 192, 3)
batch_size = 5						# Wichtig !!!			# Hier Anzahl zu vorherzusagender Masken!!!
shuffle = True
#epochs = 5



'''
# Erstellen des Trainingsnamen und Ordners
note = 'standard_tf'
architecture = 'U_Net'
model_name = '{}_voc_2012_Seg_{}_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'), architecture, note)
os.makedirs('/home/lukas/Dokumente/Trainings/{}'.format(model_name))

#-----------------------------------------------------------------------------------------------------------------------------------
'''



# Klasse Datengenerator-------------------------------------------------------------------------------------------------------------
class data_generator(tf.keras.utils.Sequence):


    def __init__(self, pic_path, mask_path, batch_size, shuffle):
        
        self.pic_path = pic_path				# Initialisierung
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.shuffle = shuffle


        pic_list = []
        mask_list = []

        for root, _, files in os.walk(pic_path):		# Einlesen aller Bildernamen
            for filename in files:
            	pic_list.append(os.path.join(root, filename))



        for root, _, files in os.walk(mask_path):		# Einlesen aller Maskennamen
            for filename in files:	
                mask_list.append(os.path.join(root, filename))


        if self.shuffle == True:
          tmp = list(zip(pic_list, mask_list))			# Shufflen der Listen
          random.shuffle(tmp)
          pic_list, mask_list = zip(*tmp)
        #print("pic_list[0]:", pic_list[0])
        #print("mask_list[0]:", mask_list[0])
        #print("Anzahl Trainingsbilder:", len(pic_list))
        #print("Anzahl Trainingsmasken:", len(mask_list))
        self.pic_list =pic_list#[5]					# Init geshuffelte Bilderliste
        self.mask_list = mask_list#[5]				# Init geshuffelte Maskenliste

        print("pic_list[0]:", pic_list[0])
        print("mask_list[0]:", mask_list[0])
        print("Anzahl Trainingsbilder:", len(pic_list))
        print("Anzahl Trainingsmasken:", len(mask_list))


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
train_gen = data_generator(val_pic_path, val_mask_path, batch_size, shuffle)

x_val, y_val = train_gen.__getitem__(0)
print('x_batch shape:', x_val.shape)
print('y_batch shape:', y_val.shape)
#print('x_batch', x_val)
#print('y_batch', y_val)

#validation_generator = [x_val, y_val]
#print('val_gen', type(validation_generator))
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




# Definieren der Kostenfunktion (Kreuzentropie + dice-loss)-------------------------------------------------------------------------------
def bce_dice_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def weighted_crossentropy(y_true, y_pred):
    weights = np.array([71/1442, 71/90, 71/79, 71/103, 71/72, 71/96, 71/74, 71/127, 71/119, 71/123, 71/71, 71/75, 71/128, 71/79, 71/76, 71/446, 71/85, 71/57, 71/90, 71/84, 71/74])
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss
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















PATH=input('Bitte geben sie den Ordnernamen des zu ladendne Modells ein:')
train_path = '/home/lukas/Dokumente/Trainings/'
save_model_path = []
save_model_path.append(os.path.join(train_path, "{}/weights.hdf5".format(PATH)))
print(save_model_path)




# Laden des Modells-------------------------------------------------------------------------------------------------------------------------
model = models.load_model(save_model_path[0], custom_objects={'bce_dice_loss': bce_dice_loss,
                                                           'dice_loss': dice_loss,
                                                            'weighted_crossentropy':weighted_crossentropy})


print("Bilder:",val_pic_path)
print("Masken:",val_mask_path)
#validation_generator = data_generator(val_pic_path, val_mask_path, batch_size)
#validation_generator = 
#print("validation_generator_shape", type(validation_generator))
#print("validation_generator_shape", validation_generator)



# Vorhersagen mit dem besten Model berechnen
predictions = model.predict(
    x_val,
    max_queue_size=128,
    workers=12, 
    use_multiprocessing=True)

print('Vorhersagen', predictions.shape)


'''
np_array = x_val[0,:,:,:]
plt.imshow(np_array)				# Ausgeben des Validierungsbildes
plt.show()
  
  
  
#encoded_np_array = np.load(np_mask_list[575])		# Laden einer Trainingsmaske vor Manipulation
encoded_np_array = y_val[0,:,:,:]	
decoded_np_array = encoded_to_np_rgb(encoded_np_array)	# Decoden der Trainingsmaske vor Manipulation

plt.imshow(decoded_np_array)				# Ausgeben der Trainismaskev or Manipulation
plt.show()


#encoded_np_array = np.load(np_mask_list[575])		# Laden einer Trainingsmaske vor Manipulation
encoded_np_array = predictions[0,:,:,:]	
decoded_np_array = encoded_to_np_rgb(encoded_np_array)	# Decoden der Trainingsmaske vor Manipulation

plt.imshow(decoded_np_array)				# Ausgeben der Trainismaskev or Manipulation
plt.show()
'''







# Ausgabe einer Bilder mit dazugehörigen (Ground-Truth) und errechneten Masken------------------------------------------------------------------
plt.figure(figsize=(10,20))
for i in range(4):
  plt.subplot(4, 3, 3 * i + 1)
  plt.imshow(x_val[i,:,:,:])
  if i==0:
    plt.title("Val-Bilder")

  encoded_mask= y_val[i,:,:,:]	
  decoded_mask = encoded_to_np_rgb(encoded_mask)

  plt.subplot(4, 3, 3 * i + 2)
  plt.imshow(decoded_mask)
  if i==0:
    plt.title("Val-Masken")

  encoded_pred = predictions[i,:,:,:]	
  decoded_pred = encoded_to_np_rgb(encoded_pred)
  
  plt.subplot(4, 3, 3 * i + 3)
  plt.imshow(decoded_pred)
  if i==0:
    plt.title("Vorhersagen")

plt.suptitle("Validierungs - Bilder,Masken und vorhergesagte Masken")
plt.show()








'''

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






def weighted_crossentropy(y_true, y_pred):
    weights = np.array([71/1442, 71/90, 71/79, 71/103, 71/72, 71/96, 71/74, 71/127, 71/119, 71/123, 71/71, 71/75, 71/128, 71/79, 71/76, 71/446, 71/85, 71/57, 71/90, 71/84, 71/74])
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss







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
def weighted_crossentropy(y_true, y_pred):
    weights = np.array([71/1442, 71/90, 71/79, 71/103, 71/72, 71/96, 71/74, 71/127, 71/119, 71/123, 71/71, 71/75, 71/128, 71/79, 71/76, 71/446, 71/85, 71/57, 71/90, 71/84, 71/74])
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss
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










