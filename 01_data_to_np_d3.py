#################################################################################################################################
#Image Segmentation with "VOC 2012" Dataset######################################################################################
#################################################################################################################################



# Installation benötiger Packeges und Toolboxen----------------------------------------------------------------------------------
import os
import glob
import zipfile
import functools

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


# Testausgabe zum Debuggen
print('\nInstallation packages erfolgleich!')
#--------------------------------------------------------------------------------------------------------------------------------



# Zuweisung von Ordnerpfaden-----------------------------------------------------------------------------------------------------
# Gewünschte Competition festlegen
competition_name = 'VOC2012'



# Festlegen von Pfaden
img_dir = os.path.join(competition_name, "JPEGImages")					# Pfad der jpg Bilder
mask_dir = os.path.join(competition_name, "SegmentationClass")				# Pfad der png Masken

train_txt_dir = os.path.join(competition_name, "Segmentation/train.txt")		# Pfad der txt-Datei mit Trainingsnamen
val_txt_dir = os.path.join(competition_name, "Segmentation/val.txt")			# Pfad der txt-Datei mit Validierungsnamen


np_image_dir = os.path.join(competition_name, "np_Images")				# Pfad für Speicherung aller Bilder im Ordner np_Images
np_SegmentationClass_dir = os.path.join(competition_name, "np_SegmentationClass")	# Pfad für Speicherung aller Masken im Ordner np_SegmentationClass


np_train_pic_dir = os.path.join(competition_name, "np_train_Images")			# Pfad für Speicherung Trainingsbilder
np_train_mask_dir = os.path.join(competition_name, "np_train_SegmentationClass")	# Pfad für Speicherung Trainingsmasken
np_val_pic_dir = os.path.join(competition_name, "np_val_Images")			# Pfad für Speicherung Validierunsbilder
np_val_mask_dir = os.path.join(competition_name, "np_val_SegmentationClass")		# Pfad für Speicherung Validierungsmasken



#trainval_txt_dir = os.path.join(competition_name, "Segmentation/trainval.txt")

# Ausgabe der Festgelegten Pfade
#print(img_dir)
#print(train_txt_dir)
#print(np_image_dir)
#print(np_SegmentationClass_dir)




# Einselsen der Trainingsbildernamen und Trainingsmaskennamen
f_train = open(train_txt_dir, 'r')
train_txt2 = f_train.readlines()
f_train.close()

train_txt = []
train_mask_txt = []
store_train_pic_txt = []		# Bei Verwendung Speicherung von Trainingsbildern in np_Images
store_train_mask_txt = []		# Bei Verwendung Speicherung von Trainingsmasken  in np_SegmentationClass
store_train_pic_dir_txt = []		# Bei Verwendung Speicherung von Trainingsbildern in np_train_Images
store_train_mask_dir_txt = []		# Bei Verwendung Speicherung von Trainingsmasken  in np_train_SegmentationClass

for img_idt in train_txt2:
  train_txt.append(os.path.join(img_dir, "{}.jpg".format(img_idt[:11]))) 
  train_mask_txt.append(os.path.join(mask_dir, "{}.png".format(img_idt[:11])))
  store_train_pic_txt.append(os.path.join(np_image_dir, '{}.npy'.format(img_idt[:11])))
  store_train_mask_txt.append(os.path.join(np_SegmentationClass_dir, '{}.npy'.format(img_idt[:11])))
  store_train_pic_dir_txt.append(os.path.join(np_train_pic_dir, '{}.npy'.format(img_idt[:11])))
  store_train_mask_dir_txt.append(os.path.join(np_train_mask_dir, '{}.npy'.format(img_idt[:11])))
  
#print('\nstore_mask_txt:', store_train_mask_txt)
#print('\n')

# Ausgabe Anzahl und Format Trainingsbilder
num_train_examples = len(train_txt)
print("\nAnzahl an Trainingsbildern: {}" .format(num_train_examples))
#print("Format eines Trainingsbildes: {}" .format(train_txt[0]))
print("Anzahl an Trainingsmasken: {}".format(len(train_mask_txt)))
#print("Format einer Trainingsmaske: {}".format(train_mask_txt[0]))
print("Anzahl an Traininsmasken als np.array: {}".format(len(store_train_mask_txt)))
print("Format einer Traininsmaske als np.array: {}".format(store_train_mask_dir_txt[0]))
#print("Format einer Trainingsmaske als np.array: {}".format(store_train_mask_txt[0]))


# Einlesen der Validierungsbildernamen und Validierungsmaskennamen
f_val = open(val_txt_dir, 'r')
val_txt2 = f_val.readlines()
f_val.close()

val_txt = []
val_mask_txt = []
store_val_pic_txt = []			# Bei Verwendung Speicherung von Bildern in np_Images
store_val_mask_txt = []			# Bei Verwendung Speicherung von Bildern in np_SegmentationClass
store_val_pic_dir_txt = []		# Bei Verwendung Speicherung von Validierungsbildern in np_val_Images
store_val_mask_dir_txt = []		# Bei Verwendung Speicherung von Validierungsmasken np_val_SegmentationClass


for img_idt in val_txt2:
  val_txt.append(os.path.join(img_dir, "{}.jpg".format(img_idt[:11]))) 
  val_mask_txt.append(os.path.join(mask_dir, "{}.png".format(img_idt[:11])))
  store_val_pic_txt.append(os.path.join(np_image_dir, "{}.npy".format(img_idt[:11])))
  store_val_mask_txt.append(os.path.join(np_SegmentationClass_dir, '{}.npy'.format(img_idt[:11])))
  store_val_pic_dir_txt.append(os.path.join(np_val_pic_dir, '{}.npy'.format(img_idt[:11])))
  store_val_mask_dir_txt.append(os.path.join(np_val_mask_dir, '{}.npy'.format(img_idt[:11])))

  

# Ausgabe Anzahl und Format Trainingsbilder
num_val_examples = len(val_txt)
print("\nAnzahl an Validierungsbildern: {}" .format(num_val_examples))
#print("Format eines Validierungsbildes: {}" .format(val_txt[0]))
print("Anzahl an Validierungsmasken: {}".format(len(val_mask_txt)))
#print("Format einer Validierungsmaske: {}".format(val_mask_txt[0]))
print("Anzahl an Validierungsmasken als np.array: {}".format(len(store_val_mask_txt)))
print("Format einer Validierungsmaske als np.array: {}".format(store_val_mask_dir_txt[0]))
#print("Format einer Validierungsmaske als np.array: {}".format(store_val_mask_txt[0]))
# Festlegen der späteren Dimensionen des np.arrays------------------------------------------------------------------------------
img_size = (192,192,3)

#-------------------------------------------------------------------------------------------------------------------------------


# Default Farbpallette für den VOC-2012 Datensatz-------------------------------------------------------------------------------
color_dict = {0:[0, 0, 0], 1:[128, 0, 0], 2:[0, 128, 0], 3:[128, 128, 0], 4:[0, 0, 128], 5:[128, 0, 128],
                      6:[0, 128, 128], 7:[128, 128, 128], 8:[64, 0, 0], 9:[192, 0, 0], 10:[64, 128, 0],
                      11:[192, 128, 0], 12:[64, 0, 128], 13:[192, 0, 128], 14:[64, 128, 128], 15:[192, 128, 128],
                      16:[0, 64, 0], 17:[128, 64, 0], 18:[0, 192, 0], 19:[128, 192, 0], 20:[0, 64, 128]}





# Funktionen zum Umwandeln und Speichern von .jpg und .png Bildern--------------------------------------------------------------

def read_png_to_np_array(png_list, store_dir ,img_size, colormap=color_dict ,encode=False):
  #Funktion zum Umwandeln von png_Bildern in ein np.array
  
  j=0
  for i in png_list:
    im = Image.open(i)				# Öffnen des i-ten Bildes entsprechend png_list
    imr = im.resize(img_size[:2], Image.NEAREST)# Resizen des Bildes
    imrc = imr.convert('RGB')			# Konvertieren zu RGB Format
    
    
    np_png_array = np.array(imrc)		# Umwandeln in ein np.array
    
    if encode == True:
      num_classes = len(colormap)
      shape = np_png_array.shape[:2]+(num_classes,)
      encoded_image = np.zeros(shape, dtype=np.int8)
      
      #print('shape', shape)
      #print('encoded_image', encoded_image.shape)
      for k, cls in enumerate(colormap):
        encoded_image[:,:,k] = np.all(np_png_array.reshape( (-1,3) ) == colormap[k], axis=1).reshape(shape[:2]) 
        
      np_png_array = encoded_image
    
    np.save(store_dir[j], np_png_array)	# Speichern des np.arrays mit Ordner/Name.npy nach store_dir[i]
    j=j+1  
    
  print('\nnp_png_arrray (Masken) shape', np_png_array.shape) # Ausgabe der Form des letzen np.arrays e.g. (256, 256, 3)
  print('np_png_array (Masken) type', type(np_png_array))     # Ausgabe des Datentysp:  <type <class 'numpy.ndarray'>
  return None





def read_jpg_to_np_array(jpg_list, store_dir, img_size):
  #Funktion zum Umwandeln von jpg_Bildern in ein np.array

  j=0
  for i in jpg_list:
    im = Image.open(i)				# Öffnen des i-ten Bildes entsprechend jpg_list
    imr = im.resize(img_size[:2], Image.NEAREST)# Resizen des Bildes
    
    np_jpg_array = np.array(imr)		# Umwandeln in ein np.array
    np_jpg_array = np_jpg_array/255.
    np.save(store_dir[j], np_jpg_array)		# Speichern des np.arrays mit mit Ordner/Name.npy nach store_dir[i]
    j=j+1

  
  print('\nnp_jpg_arrray (Bilder) shape', np_jpg_array.shape) # Ausgabe der Form des letzen np.arrays e.g. (256, 256, 3)
  print('np_jpg_array (Bilder) type', type(np_jpg_array))     # Ausgabe des Datentyps: type <class 'numpy.ndarray'>



  return None

'''
# Aufrufen der Umwandungsfunktionen mit Masken und Bildern (jeweils Train und Val)---------------------------------------------------


# Für Speicherung aller Bilder in np_Images und aller Masken in np_SegmentationClass
#read_png_to_np_array(val_mask_txt , store_val_mask_txt ,img_size, encode=True)
#read_jpg_to_np_array(val_txt, store_val_pic_txt, img_size)
#read_png_to_np_array(train_mask_txt , store_train_mask_txt ,img_size, encode=True)
#read_jpg_to_np_array(train_txt, store_train_pic_txt, img_size)


# Für Speicherung von Traingsbilder in np_train_Images, Trainingsmasken in np_train_SegmentationClass
# und Validierungsbiler in np_val_Images, Validierungsmasken in np_val_Images
read_png_to_np_array(val_mask_txt , store_val_mask_dir_txt ,img_size, encode=True)		# Umwandlung Validierungsmasken
read_jpg_to_np_array(val_txt, store_val_pic_dir_txt, img_size)					# Umwandlung Validierungsbilder
read_png_to_np_array(train_mask_txt , store_train_mask_dir_txt ,img_size, encode=True)		# Umwandlung Trainingsmasken
read_jpg_to_np_array(train_txt, store_train_pic_dir_txt, img_size)				# umwandlung Trainingsbilder


print("\nUmwandeln aller Bilder und Masken (jeweils Trian und Val) erfolgreich!")


print("\nstore_dir", store_train_mask_txt[0])
'''





# Laden, decoden und ausgeben eines np.arrays (encoded)-------------------------------------------------------------------------------

def encoded_to_np_rgb(onehot, colormap = color_dict):
    # Funktion zum encoden von Trainingsmasken 
    # (h, w, 21), Null/Eins je Klasse 	   ->      (h, w, 3), RGB-Wert je Pixel
 
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


encoded_np_array = np.load(store_train_mask_dir_txt[275])	# Laden einer Trainingsmaske
decoded_np_array = encoded_to_np_rgb(encoded_np_array)		# Decoden der Trainingsmaske


plt.imshow(decoded_np_array)					# Ausgeben der Trainismaske
plt.show()


# Laden und ausgeben eines np.arrays (nicht encoded)----------------------------------------------------------------------------------

np_array = np.load(store_train_pic_dir_txt[275])		# Laden eines Traingsbilds

#print("\nnp_array", np_array[0:100])

plt.imshow(np_array)						# Ausgeben des Trainingsilds
plt.show()




