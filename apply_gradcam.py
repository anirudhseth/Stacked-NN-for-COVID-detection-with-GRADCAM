
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt




class GradCAM:
	def __init__(self, model, classIdx, layerName):
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName

	def compute_heatmap(self, image, eps=1e-8):
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output, 
				self.model.output])
		with tf.GradientTape() as tape:
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		grads = tape.gradient(loss, convOutputs)

		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))

		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")

		return heatmap

	def overlay_heatmap(self, heatmap, image, alpha=0.3,
		colormap=cv2.COLORMAP_JET):
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		return (heatmap, output)


argsimg='test2/2e5dc881-3eef-4f78-96f8-2e240bbede2a.png'    ### filepath from text file 
label= 1  #store the true label here

model=tf.keras.models.load_model('model1')			### trained model path



orig = cv2.imread(argsimg)
resized = cv2.resize(orig, (224, 224))

image = cv2.imread(argsimg)
h, w, c = image.shape
image = image[int(h/6):, :]
image = cv2.resize(image, (224,224))

image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

preds = model.predict(image)
i = np.argmax(preds[0])


cam = GradCAM(model, label,layerName='block6_conv2')			## layer name  . this should be the last conv layer so no change but we can visulaize other layers
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
ax3.set_title('GradCam')

# plt.savefig('fd')