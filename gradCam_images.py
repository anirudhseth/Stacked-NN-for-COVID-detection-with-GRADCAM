# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from apply_gradcam import *
import os
import numpy as np
import pandas as pd


folder = 'E:/deepLearning/data'
wfolder = './figure'
model_name = 'model1'
model=tf.keras.models.load_model(model_name)


c_name = ['pID', 'fname', 'diagnose', 'extra', 'add extra', 'source']
df_test = pd.read_csv("test_split_v5.txt", header= None,  names = c_name) #sep =',',
df_test = df_test.loc[:,['fname', 'diagnose','source']]
df_test['diagnose'] = np.where(df_test['diagnose'] == 'normal', 0, df_test['diagnose'])
df_test['diagnose'] = np.where(df_test['diagnose'] == 'pneumonia', 1, df_test['diagnose'])
df_test['diagnose'] = np.where(df_test['diagnose'] == 'COVID-19', 2, df_test['diagnose'])


df_test.head(2)


# %%
df_test['diagnose'].value_counts()


# %%
df_test_normal = df_test[df_test['diagnose'] == 0].sample(10)
df_test_pneumonia = df_test[df_test['diagnose'] == 1].sample(10) 
df_test_covid = df_test[df_test['diagnose'] == 2].sample(10) 

df_test_normal.reset_index(inplace=True)
df_test_pneumonia.reset_index(inplace=True)
df_test_covid.reset_index(inplace=True)

# %%
def show_pic(df, src_folder, des_folder):
    for irow in range(len(df)):
        argsimg = src_folder + '/' + df.loc[irow, 'source'] + '/' + df.loc[irow, 'fname']
        #print(argsimg)
        label = df.loc[irow, 'diagnose']
        orig = cv2.imread(argsimg)
        print("------------------")
        try:
            resized = cv2.resize(orig, (224, 224))
            image = cv2.imread(argsimg)
            h, w, c = image.shape
            image = image[int(h/6):, :]
            image = cv2.resize(image, (224,224))

            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)

            preds = model.predict(image)
            i = np.argmax(preds[0])
            ## layer name  . this should be the last conv layer so no change but we can visulaize other layers
            cam = GradCAM(model, label, layerName='block6_conv2') 
            heatmap = cam.compute_heatmap(image)

            heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
            (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

            fig, (ax1,ax3) = plt.subplots(1, 2)
            fig.set_size_inches(9, 5)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.imshow(orig)
            ax1.set_title('Orignal Image')

            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.imshow(output)
            ax3.set_title('GradCam' + str(label))
            saveFilePath = des_folder + str(df.loc[irow, 'diagnose']) + '_' + df.loc[irow, 'source'] + '_' + df.loc[irow, 'fname'].split(sep = '.')[0] + '.png'

            plt.savefig(saveFilePath)    
            plt.close(fig)  
        except:
            print("wrong: ", df.loc[irow, 'fname'])


# %%
show_pic(df_test_normal, folder, wfolder + '/'+ model_name + '_')
show_pic(df_test_pneumonia, folder, wfolder + '/'+ model_name + '_')
show_pic(df_test_covid, folder, wfolder + '/'+ model_name + '_')


