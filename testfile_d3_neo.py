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
img_dir = os.path.join(competition_name, "JPEGImages")
mask_dir = os.path.join(competition_name, "SegmentationClass")

train_txt_dir = os.path.join(competition_name, "Segmentation/train.txt")
val_txt_dir = os.path.join(competition_name, "Segmentation/val.txt")
#trainval_txt_dir = os.path.join(competition_name, "Segmentation/trainval.txt")

# Ausgabe der Festgelegten Pfade
#print(img_dir)
#print(train_txt_dir)




# Einselsen der Trainingsbildernamen und Trainingsmaskennamen
f_train = open(train_txt_dir, 'r')
train_txt2 = f_train.readlines()
f_train.close()

train_txt = []
train_mask_txt = []
for img_idt in train_txt2:
  train_txt.append(os.path.join(img_dir, "{}.jpg".format(img_idt[:11]))) 
  train_mask_txt.append(os.path.join(mask_dir, "{}.png".format(img_idt[:11])))
# Ausgabe Anzahl und Format Trainingsbilder
num_train_examples = len(train_txt)
print("\nAnzahl an Trainingsbildern: {}" .format(num_train_examples))
print("Format eines Trainingsbildes: {}" .format(train_txt[0]))
print("Anzahl an Trainingsmasken: {}".format(len(train_mask_txt)))
print("Format einer Trainingsmaske: {}".format(train_mask_txt[0]))



# Einlesen der Validierungsbildernamen und Validierungsmaskennamen
f_val = open(val_txt_dir, 'r')
val_txt2 = f_val.readlines()
f_val.close()

val_txt = []
val_mask_txt = []
for img_idt in val_txt2:
  val_txt.append(os.path.join(img_dir, "{}.jpg".format(img_idt[:11]))) 
  val_mask_txt.append(os.path.join(mask_dir, "{}.png".format(img_idt[:11])))

# Ausgabe Anzahl und Format Trainingsbilder
num_val_examples = len(val_txt)
print("\nAnzahl an Validierungsbildern: {}" .format(num_val_examples))
print("Format eines Validierungsbildes: {}" .format(val_txt[0]))
print("Anzahl an Validierungsmasken: {}".format(len(val_mask_txt)))
print("Format einer Validierungsmaske: {}".format(val_mask_txt[0]))










# Ausgabe einer Auswahl an Bildern--------------------------------------------------------------------------------------------

# Anzahl der anzuzeigenden Bildern
display_num = 4

r_choices = np.random.choice(num_train_examples, display_num)
print("Folgende Bilder werden ausgegebn:", r_choices)
# Ploteinstellungen
plt.figure(figsize=(10, 15))
for i in range(0, display_num * 2, 2):
  img_num = r_choices[i // 2]
  x_pathname = train_txt[img_num]
  y_pathname = train_mask_txt[img_num]
  
  plt.subplot(display_num, 2, i + 1)
  plt.imshow(mpimg.imread(x_pathname))
  plt.title("Original Image")
  
  example_labels = Image.open(y_pathname)
  label_vals = np.unique(example_labels)
  
  plt.subplot(display_num, 2, i + 2)
  plt.imshow(example_labels)
  plt.title("Masked Image")  
  
plt.suptitle("Beispiele für Trainingsbilder und deren Masken aus dem 'VOC2012-Datensatz'")
plt.show()










# Festlegung von Parametern----------------------------------------------------------------------------------------------------

img_shape = (128, 128, 3)		# Muss durch 32 teilbar sein
batch_size = 5				# Größe Batch
epochs = 1				# Anzahl an Trainingsepochen

# Erstellen des Trainingsnamen und Ordners
note = 'standard'
architecture = 'U_Net'
model_name = '{}_VOC2012_Seg_{}_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'), architecture, note)
os.makedirs('/home/lukas/Dokumente/Trainings/{}'.format(model_name))







# Funktionen zum Preprocessen von Eingangsbildern-----------------------------------------------------------------------------

def shift_img(output_img, label_img, width_shift_range, height_shift_range):
  """Funktion zum zufälligen shiften von Trainingsbildern und dazugehörigen Labeln"""
  if width_shift_range or height_shift_range:
      if width_shift_range:
        width_shift_range = tf.random_uniform([], 
                                              -width_shift_range * img_shape[1],
                                              width_shift_range * img_shape[1])
      if height_shift_range:
        height_shift_range = tf.random_uniform([],
                                               -height_shift_range * img_shape[0],
                                               height_shift_range * img_shape[0])
      # Translate both 
      output_img = tfcontrib.image.translate(output_img,
                                             [width_shift_range, height_shift_range])
      label_img = tfcontrib.image.translate(label_img,
                                             [width_shift_range, height_shift_range])
  return output_img, label_img




def flip_img(horizontal_flip, tr_img, label_img):
  """Funktion zum Flippen von Trainingsbildern und dazugehörigen Labeln (50%)"""
  if horizontal_flip:
    flip_prob = tf.random_uniform([], 0.0, 1.0)
    tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                lambda: (tr_img, label_img))
  return tr_img, label_img




def _augment(img,
             label_img,			# Defaultwerte:
             resize=None,  		# Resize the image to some size e.g. [256, 256]
             scale=1,  			# Scale image e.g. 1 / 255.
             hue_delta=0,  		# Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  	# Random left right flip,
             width_shift_range=0,  	# Randomly translate the image horizontally
             height_shift_range=0):  	# Randomly translate the image vertically 
  """Funktion zum Datensatz erweitern"""
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize_images(label_img, resize)
    img = tf.image.resize_images(img, resize)
  
  if hue_delta:
    img = tf.image.random_hue(img, hue_delta)
  
  img, label_img = flip_img(horizontal_flip, img, label_img)
  img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
  label_img = tf.to_float(label_img) * scale
  img = tf.to_float(img) * scale 
  print("\n")
  print("Datentyp nach augmention:", type(label_img))
  print("\n")
  return img, label_img






# (Daten-) Pipline mit tf.data einrichten-----------------------------------------------------------------------------

def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair  
  img_str = tf.read_file(fname)
  img = tf.image.decode_jpeg(img_str, channels=0)

  label_img_str = tf.read_file(label_path)
  # These are gif images so they return as (num_frames, h, w, c)
  label_img = tf.io.decode_png(label_img_str, channels=3)
  # The label image should only have values of 1 or 0, indicating pixel wise
  # object (car) or not (background). We take the first channel only. 
  #label_img = label_img[:, :, 0]
  #label_img = tf.expand_dims(label_img, axis=-1)
  return img, label_img


def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=batch_size,
                         shuffle=True):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
    assert batch_size == 1, "Batching images must be of the same size"

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  
  print('\n')
  print('Datentyp dataset', type(dataset))
  #print('Shape dataset', dataset.shape) -> error

  if shuffle:
    dataset = dataset.shuffle(num_x)
  
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size)
  return dataset










# Festlegung Parameter für die Preprocessing Funktion für Trainings- und Validierungsdaten----------------------------

tr_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 1000.,
    'hue_delta': 0.25,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)


val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 1000.,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)







# Erstellung eines Training und Validierungsdatensatzes-----------------------------------------------------------------

train_ds = get_baseline_dataset(train_txt,
                                train_mask_txt,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)

val_ds = get_baseline_dataset(train_txt,
                              train_mask_txt, 
                              preproc_fn=val_preprocessing_fn,
                              batch_size=batch_size)









# Ausgabe von TRAININGS-Trainingsbildern nach dem Preprocessing------------------------------------------------------------------

temp_ds = get_baseline_dataset(train_txt, 
                               train_mask_txt,
                               preproc_fn=tr_preprocessing_fn,
                               batch_size=5)

# Ausgabe von Trainings-Bildern nach dem Preprocessing
data_aug_iter = temp_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()


with tf.Session() as sess: 
  batch_of_imgs, label = sess.run(next_element)
  print("Datentyp der Trainingsbilder:", type(batch_of_imgs))
  print('Datentyp der Trainingsmasken:', type(label))
  print("Format der Trainingsbilder:", batch_of_imgs.shape)
  print("Format der Trainingsmasken:", label.shape)
  
  
  fig, axs = plt.subplots(3,2)
  axs[0,0].imshow(batch_of_imgs[0])
  axs[0,1].imshow(label[0])
  
  axs[1,0].imshow(batch_of_imgs[1])
  axs[1,1].imshow(label[1])
  
  axs[2,0].imshow(batch_of_imgs[2])
  axs[2,1].imshow(label[2])

  axs[0,0].set_title('Originalbilder')
  axs[0,1].set_title('Masken')
  
  
  plt.suptitle("Trainingsbilder und deren Masken aus dem 'VOC2012-Datensatz' nach dem Preprocessing")
    
  #print(batch_of_imgs[0].shape)
  #print(label[0, :, :, 0].shape)
 
  #y_true_f = tf.reshape(batch_of_imgs[0], [-1])
  
  #y_pred_f = sess.run(tf.reshape(label[0, :, :, 0], [-1]))
  #intersection = tf.reduce_sum(y_true_f * y_pred_f)
  #print(y_true_f)
  #print(intersection) 
  #print(label.shape)
  
  plt.show()

#---------------------------------------------------------------------------------------------------------------------



# Colormap voc2012----------------------------------------------------------------------------------------------------



#color_dict = {0:[0/255., 0/255., 0/255.], 1:[128/255., 0/255., 0/255.], 2:[0/255., 128/255., 0/255.], 3:[128/255., 128/255., 0/255.], 4:[0/255., 0/255., 128/255.], 5:[128/255., 0/255., 128/255.],
#                      6:[0/255., 128/255., 128/255.], 7:[128/255., 128/255., 128/255.], 8:[64/255., 0/255., 0/255.], 9:[192/255., 0/255., 0/255.], 10:[64/255., 128/255., 0/255.],
#                      11:[192/255., 128/255., 0/255.], 12:[64/255., 0/255., 128/255.], 13:[192/255., 0/255., 128/255.], 14:[64/255., 128/255., 128/255.], 15:[192/255., 128/255., 128/255.],
#                      16:[0/255., 64/255., 0/255.], 17:[128/255., 64/255., 0/255.], 18:[0/255., 192/255., 0/255.], 19:[128/255., 192/255., 0/255.], 20:[0/255., 64/255., 128/255.]}

color_dict = {0:[0, 0, 0], 1:[128, 0, 0], 2:[0, 128, 0], 3:[128, 128, 0], 4:[0, 0, 128], 5:[128, 0, 128],
                      6:[0, 128, 128], 7:[128, 128, 128], 8:[64, 0, 0], 9:[192, 0, 0], 10:[64, 128, 0],
                      11:[192, 128, 0], 12:[64, 0, 128], 13:[192, 0, 128], 14:[64, 128, 128], 15:[192, 128, 128],
                      16:[0, 64, 0], 17:[128, 64, 0], 18:[0, 192, 0], 19:[128, 192, 0], 20:[0, 64, 128]}


print('\n color_dict', color_dict[1])

class_names = {0:'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat',
                   5:'bottle', 6:'bus', 7:'car', 8:'cat', 9:'chair', 10:'cow', 11:'diningtable',
                   12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'potted-plant',
                   17:'sheep', 18:'sofa', 19:'train', 20:'tv/monitor'}


#print("Color dictionary", color_dict)
#print("Class names",class_names)



def rgb_to_onehot(rgb_image, colormap = color_dict):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    print("datentyp rgb_image", type(rgb_image))
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    encoded_image2= np.zeros( shape, dtype=np.int8 )
    #print('Shape encoded_image:',encoded_image.shape) -> (128,128,21)
    a=rgb_image.reshape( (-1,3) )
    b=tf.reshape(rgb_image, (-1,3))
    a2 = np.zeros(a.shape, dtype=np.int8)
    print("a:", a.shape)
    print("b:", b.shape)

    encoded_label_neo = np.zeros ((16384,21), dtype=np.int8)
    neo_neo = tf.reshape(rgb_image, (-1,3))
    print('neo_neo shape:', neo_neo.shape)
    print('neo_neo type:', type(neo_neo))
    print('a type:', type(a))
#   print('neo_neo Werte', neo_neo[0:10,2])

    for i, cls in enumerate(colormap):
        #test = tf.math.reduce_all(neo_neo == colormap[i], axis=1)
        #encoded_label[:,:,i] = np.all(neo_neo == colormap[i], axis=1)
        encoded_image[:,:,i] = tf.reshape(tf.keras.backend.all(tf.reshape(rgb_image, (-1,3)) == colormap[i], axis=1, keepdims=True), (128,128))
        #encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])   
        
        #temp = tf.reshape(rgb_image, (-1,3))*1000
        #temp = tf.reshape(temp, (128,128))
        #temp = tf.reshape(rgb_image, (-1,3))
        #if i == 0:
        #   print('temp shape',temp.shape)
        #encoded_label_neo[:, i] = np.all(temp == colormap[i], axis=1)
    #encoded_label = tf.reshape(encoded_label_neo, shape[:2])
    return encoded_image



def onehot_to_rgb(onehot, colormap = color_dict):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)



c = np.zeros
b = tf.reshape(label[0], (-1,3))
print(b.shape)
if b[0,:]==color_dict[0]:
  b[0, :] = 1
else:
  b[0, :] = [0,0,0]

#a = tf.math.equal(b[0,:], color_dict[0])
 
print('b=', b[0,:])
test_encode_mask = rgb_to_onehot(label[0], color_dict)
test_decode_mask = onehot_to_rgb(test_encode_mask, color_dict)


print('Format test_encode_mask:', test_encode_mask.shape)
print(test_encode_mask[48:68,48:68,0])
print('/n')
print(test_encode_mask[48:64,48:64,1])
print('/n')
print(test_encode_mask[48:64,48:64,2])
print('/n')
print(test_encode_mask[48:64,48:64,3])
print('/n')
print(test_encode_mask[48:64,48:64,4])
print('/n')
print(test_encode_mask[48:64,48:64,5])
print('/n')
print(test_encode_mask[48:64,48:64,6])
print('/n')
print(test_encode_mask[48:64,48:64,7])
print('/n')
print(test_encode_mask[48:64,48:64,8])
print('/n')
print(test_encode_mask[48:64,48:64,9])
print('/n')
print(test_encode_mask[48:64,48:64,10])
print('/n')
print(test_encode_mask[48:64,48:64,11])
print('/n')
print(test_encode_mask[48:64,48:64,12])
print('/n')
print(test_encode_mask[48:64,48:64,13])
print('/n')
print(test_encode_mask[48:64,48:64,14])
print('/n')
print(test_encode_mask[48:64,48:64,15])
print('/n')
print(test_encode_mask[48:64,48:64,16])
print('/n')
print(test_encode_mask[48:64,48:64,17])
print('/n')
print(test_encode_mask[48:64,48:64,18])
print('/n')
print(test_encode_mask[48:64,48:64,19])
print('/n')
print(test_encode_mask[48:64,48:64,20])
print('/n')







a = tf.keras.backend.flatten(test_encode_mask)
print("Format a:", a)


plt.imshow(test_decode_mask)
plt.show()



"""

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################




# Definieren der Blockstrucktur-------------------------------------------------------------------------------------------------------------

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
  inputs = layers.Input(shape=img_shape)
  # 256


with tf.name_scope('Encoder_1'):
  encoder0_pool, encoder0 = encoder_block(inputs, 32)
  # 128


with tf.name_scope('Encoder_2'):
  encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
  # 64


with tf.name_scope('Encoder_3'):
  encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
  # 32


with tf.name_scope('Encoder_4'):
  encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
  # 16


with tf.name_scope('Encoder5'):
  encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
  # 8


with tf.name_scope('Center'):
  center = conv_block(encoder4_pool, 1024)
  # center


with tf.name_scope('Decoder_1'):
  decoder4 = decoder_block(center, encoder4, 512)
  # 16


with tf.name_scope('Decoder_2'):
  decoder3 = decoder_block(decoder4, encoder3, 256)
  # 32


with tf.name_scope('Decoder_3'):
  decoder2 = decoder_block(decoder3, encoder2, 128)
  # 64


with tf.name_scope('Decoder_4'):
  decoder1 = decoder_block(decoder2, encoder1, 64)
  # 128


with tf.name_scope('Decoder_5'):
  decoder0 = decoder_block(decoder1, encoder0, 32)
  # 256


with tf.name_scope('Outputlayer'):
  outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
  #256




# Definieren des Modells-----------------------------------------------------------------------------------------------------------------
model = models.Model(inputs=[inputs], outputs=[outputs])




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



# Instanzieren des Modells
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])




# Erstellung Tensorboard-Graph und Modellspeicherung--------------------------------------------------------------------------------------
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/home/lukas/Dokumente/Trainings/{}/logs'.format(model_name))

model.summary()

save_model_path = '/home/lukas/Dokumente/Trainings/{}/weights.hdf5'.format(model_name) 

cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)





# Durchführung des Trainings-------------------------------------------------------------------------------------------------------------
history = model.fit(train_ds, 
                   steps_per_epoch=int(np.ceil(len(train_txt) / float(batch_size))),
                   epochs=epochs,
                   validation_data=val_ds,
                   validation_steps=int(np.ceil(len(val_txt) / float(batch_size))),
                   callbacks=[tensorboard, cp])





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






# Laden des Modells 
model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                           'dice_loss': dice_loss})

temp_ds2 = get_baseline_dataset(val_txt[:20], 
                                val_mask_txt[:20], #(eigentlich labels, gibt es aber für Testbilder nicht!)
                                preproc_fn=val_preprocessing_fn,
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





######################################################################################################
######################################################################################################
######################################################################################################

"""












