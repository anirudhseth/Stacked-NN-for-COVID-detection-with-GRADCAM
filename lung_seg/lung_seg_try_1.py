
""" 
Requires tensorflow > 1.1.0 and Keras > 2.0.4
I ran with latest versions of both which works fine

Model is pretrained on Montgomery and JSRT data with 
lung masks from http://www.isi.uu.nl/Research/Databases/SCR/

Lung masks have been created by two observers.

"The following definition for the lung fields is adopted: any pixel for which 
radiation passed through the lung, but not through the mediastinum, the heart, 
structures below the diaphragm, and the aorta. The vena cava superior, when visible, 
is not considered to be part of the mediastinum"

More info on JSRT dataset: http://www.isi.uu.nl/Research/Databases/SCR/data.php
"""

import sys , os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform
from matplotlib import pyplot as plt

def loadDataGeneral(df, path, im_shape):
    """ A few imgs have other dims so they get sorted out 
        Approx 6 imgs per 200

        This part loads imgs, reshapes them, 
        applies histogram equalization and normalizes.
    """

    X = []
    for item in df.iterrows():
        img = img_as_float(io.imread(path + item[1][0]))
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        if len(np.shape(img)) == 3:
            X.append(img)

    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print('### Dataset loaded')
    print('\t{}'.format(path))
    print('\t{}'.format(X.shape))
    print('\tX:{:.1f}-{:.1f}\t\n'.format(X.min(), X.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def main(argv):
    from_i , n , rel_csv_path , rel_img_path , savefile_path = argv
    from_i = int(from_i)
    n = int(n)
    

    """
    arg1: from which image in folder to start
    arg2: how many imgs to crop
    arg3: path to csv file from cwd
    arg4: path to x-ray imgs from cwd
    arg5: where to save cropped imgs

    Imgs will be saved with same name as before but with "_cropped" added extention."""
    
    # Path to csv-file. File should contain X-ray filenames as a column
    #csv_path = 'lung-segmentation-2d-master/idx_004.csv'
    csv_path = rel_csv_path
    
    # Path to the folder with images. Images will be read from path
    #path = 'lung-segmentation-2d-master/images_004/images/'
    path = rel_img_path

    #savefile_path = 'lung-segmentation-2d-master/New_cropped_imgs/'
    savefile_path = savefile_path

    print("Beginning to read file...\n")
    print("From img no %d to %d , %d imgs total\n" % (from_i, (from_i+n), n))

    df = pd.read_csv(csv_path,skiprows=from_i,chunksize=n).get_chunk()
    print("File read!\n")

    img_names = [str.split(string,'.') for string in df.values[:,0]]

    # Load data
    im_shape = (256, 256)
    X = loadDataGeneral(df, path, im_shape)

    n = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    # Pretrained 
    model_name = 'trained_model.hdf5'
    UNet = load_model(model_name)
    
    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    i = 0
    print("Beginning segmentation of %d images\n" % n)

    for xx in test_gen.flow(X, batch_size=1):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        pr = pred > 0.5

        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        
        plt.figure(figsize=(10,10))
        plt.axis('off')
        plt.imshow(pr.astype(np.int8)*img, cmap='gray', vmin=0, vmax=1)
        plt.savefig(savefile_path + img_names[i][0] + '_crop.' + img_names[i][1])

        i += 1
        if i == n:
            break

    print("Done! Images saved to\n")
    print(savefile_path)
    print("##################################\n")

if __name__ == '__main__':
    main(sys.argv[1:])
