#################################################################################################################################
#Image Segmentation with "VOC 2012" Dataset######################################################################################
#################################################################################################################################



# Installation benötiger Packeges und Toolboxen----------------------------------------------------------------------------------
import os
import glob
import zipfile
import functools
import cv2
import csv

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
np_image_dir = os.path.join(competition_name, "np_train_Images")				# Pfad zu den Trainingsbildern, bereits np_arrays
np_mask_dir = os.path.join(competition_name, "np_train_SegmentationClass")			# Pfad zu den Trainingsmasken,  bereits np_arrays

np_train_pic_aug_dir = os.path.join(competition_name, "np_train_Images_aug")			# Pfad für Speicherung Trainingsbilder (nach dataAugmention)
np_train_mask_aug_dir = os.path.join(competition_name, "np_train_SegmentationClass_aug")	# Pfad für Speicherung Trainingsmasken (nach dataAugmention)



# Ausgabe der Festgelegten Pfade
print("\nOrdnerpfade:")
print(np_image_dir)
print(np_mask_dir)
print(np_train_pic_aug_dir)
print(np_train_mask_aug_dir)
print("Ende Ordnerpfade")
#---------------------------------------------------------------------------------------------------------------------------------












# Einlesen der zu manipulierenden np_train_images und np_train_masks--------------------------------------------------------------

np_pic_list = os.listdir(np_image_dir)				# Liste aller np_pic_arrays
np_mask_list = os.listdir(np_mask_dir)				# Liste aller np_mask_arrays 
np_pic_list_aug = []						# Dateinamen aller np_pic_aug_arrays  (Dateipfad + Name) zum Speichern
np_mask_list_aug = []						# Dateinamen aller np_mask_aug_arrays (Dateipafd + Name) zum Speichern

#np_pic_list_aug.append(os.path.join(np_train_pic_aug_dir, "{}_aug.npy".format(img_idt[:11])))
#  train_txt.append(os.path.join(img_dir, "{}.jpg".format(img_idt[:11]))) 
j=0
for img_idt in np_pic_list:    
  np_pic_list_aug.append(os.path.join(np_train_pic_aug_dir, "{}_aug.npy".format(img_idt[:11])))
  np_pic_list[j] = os.path.join(np_image_dir, np_pic_list[j])	# Liste aller np_pic_arrays mit vollständigem Pfad
  #np_mask_list[j] = os.path.join(np_mask_dir, np_mask_list[j])  # Liste aller np_mask_arrays mit vollständigem Pfad
  j=j+1


j=0
for img_idt in np_mask_list:
  np_mask_list_aug.append(os.path.join(np_train_mask_aug_dir, "{}_aug.npy".format(img_idt[:11])))
  np_mask_list[j] = os.path.join(np_mask_dir, np_mask_list[j])  # Liste aller np_mask_arrays mit vollständigem Pfad
  j=j+1


print("\nAnzahl np_pic_arrays: {}".format(len(np_pic_list)))

print("Anzahl np_mask_arrays: {}".format(len(np_mask_list)))
print("Beispiel eines Bildpfades:", np_pic_list[463])
print("Beispiel eines Maskenpfades:", np_mask_list[463])
print("Dimension eines np_pic_arrays:", np.load(np_pic_list[0]).shape)
print("Dimension eines np_mask_arrays:", np.load(np_mask_list[0]).shape)

print("\nBeispiel eines Bildpfades zum Speichern nach dataAug:", np_pic_list_aug[463])
print("Beispiel eines Maskenpfades zum Speichern nach dataAug:", np_mask_list_aug[463])
#---------------------------------------------------------------------------------------------------------------------------------












# Festlegen der späteren Dimensionen des np.arrays-------------------------------------------------------------------------------
img_size = (192,192,3)		# Hier überflüssig, da mit bereits mit (in der gewünschten Grüße erzeugten) np_arrays 
				# gearbeitet wird
#--------------------------------------------------------------------------------------------------------------------------------

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
#---------------------------------------------------------------------------------------------------------------------------------












# Funktionen für die DataAugmention---------------------------------------------------------------------------------------------------

def gauss_noise(image):
  # Funktion zum Addieren von Gausschem Rauschen auf die Trainingsbilder
    row,col,ch= image.shape
    mean = 0
    var = 0.005
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noise = image + gauss
    #print(noise.shape)
    i=0
    j=0
    k=0  
    # Abfangen von Werten größer 1, kleiner 0   
    for a in range(col):
      i=0
      for b in range(row):  
        k=0
        for c in range(ch):
          if noise[i,j,k] < 0:
            noise[i,j,k] = 0  
          if noise[i,j,k] > 1:
            noise[i,j,k] =1
          k=k+1
        i=i+1
      j=j+1
    return noise


def manipulate_np_array(pic_list, store_pic ,mask_list, store_mask):
  # Funktion für die Datensatzerweiterung
  # Funktionsteil zur data_aug für die Bilder------------------
  i=0
  for ph in pic_list:
  
    np_array = np.load(ph)						# Laden des zu manipulierenden Trainingsbildes
    #print("Typ original", type(np_array))				# Ausgabe Datentyp des Trainingsbildes
    #print("original:", np_array[:50])					# Ausgabe einiger Werte des Trainingsbildes 

    np_array_rot = np.rot90(np.fliplr(np_array))			# Spiegelung und Rotation des Trainingsbildes
    np_array_aug = gauss_noise(np_array_rot)				# Addieren eines Gaussschen Rauschens auf das Trainingsbild 
  
    #print("augmention:", np_array_aug[:50])				# Ausgabe einer Werte des manipulierten Trainingsbildes
    #plt.imshow(np_array_aug)						# Ausgeben des Trainingsilds
    #plt.show()

    np.save(store_pic[i], np_array_aug)					# Speichern des np.arrays mit Ordner/Name.npy nach store_dir[i]
    i=i+1
  
    # Funktionsteil zur data_aug für die Masken------------------
  j=0
  for ph in mask_list:  
    encoded_np_array = np.load(ph)					# Laden einer Trainingsmaske
    encoded_np_array_aug = np.rot90(np.fliplr(encoded_np_array))	# Spiegelung und Rotation der Trainingsmaske
    #decoded_np_array = encoded_to_np_rgb(encoded_np_array_aug)		# Decoden der Trainingsmaske
    #plt.imshow(decoded_np_array)					# Ausgeben der Trainismaske
    #plt.show()

    np.save(store_mask[j], encoded_np_array_aug)			# Speichern des np.arrays mit Ordner/Name.npy nach store_dir[i]
    j=j+1

''' Aufrufen der dataAugmention Funktion für Trainingsbilder und Trainigsmasken
#manipulate_np_array(np_pic_list, np_pic_list_aug ,np_mask_list, np_mask_list_aug)	
'''
#----------------------------------------------------------------------------------------------------------------------------------






# Laden und ausgeben eines np.arrays (Trainingsmaske, nicht encoded)---------------------------------------------------------------

np_array = np.load(np_pic_list[575])			# Ausgabe Testbild vor Manipulation
plt.imshow(np_array)
plt.show()

np_array = np.load(np_pic_list_aug[575])		# Ausgabe Testbild nach Manipulation
plt.imshow(np_array)
plt.show()





# Laden und ausgeben eines np.arrays (Trainingsmaske, encoded)---------------------------------------------------------------------

encoded_np_array = np.load(np_mask_list[575])		# Laden einer Trainingsmaske vor Manipulation
decoded_np_array = encoded_to_np_rgb(encoded_np_array)	# Decoden der Trainingsmaske vor Manipulation

plt.imshow(decoded_np_array)				# Ausgeben der Trainismaskev or Manipulation
plt.show()


encoded_np_array = np.load(np_mask_list_aug[575])	# Laden einer Trainingsmaske nach Manipulation
decoded_np_array = encoded_to_np_rgb(encoded_np_array)	# Decoden der Trainingsmaske nach Manipulation

plt.imshow(decoded_np_array)				# Ausgeben der Trainismaske nach Manipulation
plt.show()





















