import numpy as np
import struct
import timeit
import cv2

class Helpers():
	@staticmethod
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def grad_sigmoid(x):
		return x * (1 - x)

	@staticmethod
	def tanh(x):
		return np.tanh(x)

	@staticmethod
	def grad_tanh(x):
		return 1 - x * x

class Layer():
	def __init__(self, n_inputs, n_neurons):
		self.weights = (2 * np.random.random((n_inputs, n_neurons)) - 1) / 10

	def adjust(self, adjustment, learning_rate, old_adj): 
		self.weights -= (learning_rate * adjustment) + (0.05 * old_adj)

class NeuralNetwork():
	def __init__(self, layers, dataset):
		self.layers= layers
		self.dataset= dataset
		self.samini= np.array([np.zeros(l.weights.shape) for l in layers])
		self.learning_rate= 0.01
		self.layers_output= []
		self.old_adjustment= np.zeros(len(self.layers))

	def feedforward(self,inputs):
		self.layers_output= []
		self.layers_output.append(Helpers.tanh(np.dot(inputs, self.layers[0].weights))) 

		for i in xrange(len(self.layers)-1):
			self.layers_output.append(Helpers.sigmoid(np.dot(self.layers_output[i], self.layers[i+1].weights)))

		return self.layers_output
    

	def backpropagate(self,images,labels):
		error= [None] * len(self.layers)
		delta= [None] * len(self.layers)
		adjustment= [None] * len(self.layers)
		
		error[len(self.layers)-1]= (labels - self.layers_output[len(self.layers_output)-1])
		delta[len(self.layers)-1]= ((-1 * error[len(self.layers)-1] * Helpers.grad_sigmoid(self.layers_output[len(self.layers_output)-1])).reshape((1,1)))
		adjustment[len(self.layers)-1]= (self.layers_output[len(self.layers_output)-2].T.dot(delta[len(self.layers)-1]))
		for i in (reversed(xrange(1,len(self.layers_output)-1))):  
			error[i]=delta[i+1].dot(self.layers[i+1].weights.T)
			delta[i]=error[i] * Helpers.grad_sigmoid(self.layers_output[i])
			adjustment[i]= self.layers_output[i-1].T.dot(delta[i])
		
		error[0]= delta[1].dot(self.layers[1].weights.T)
		delta[0]= (error[0] * Helpers.grad_tanh(self.layers_output[0]))
		adjustment[0]= images.T.dot(delta[0])
           
		for i in xrange(len(self.layers)):   
			self.layers[i].adjust(adjustment[i],self.learning_rate,self.old_adjustment[i]) 
        
		self.old_adjustment= adjustment
		self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * 0.001)))


	def run(self,iterations):
		for j in xrange(1,iterations):
			error=0.0
			for i in xrange(len(self.dataset[0])):
				error+= (self.feedforward((self.dataset[0][i]).reshape((1,len(self.dataset[0][i].T))))[len(self.layers_output)-1] - self.dataset[1][i]) ** 2 * 0.5
				self.backpropagate((self.dataset[0][i]).reshape((1,len(self.dataset[0][i].T))),(self.dataset[1][i]).reshape((1,len(self.dataset[1][i].T))))
			if (j%10==0):
				print 'Training iteration %s , error : %s' % (j,error/len(self.dataset[0]))

	def test(self,testing_set):
		error=0.0
		error_rounded= 0.0
		for i in xrange(len(testing_set[0])):
			output= self.feedforward(testing_set[0][i])
			error+= (output[len(self.layers_output)-1]-testing_set[1][i] ) ** 2 * 0.5;
			error_rounded+= (round((9*output[len(self.layers_output)-1])-(9*testing_set[1][i]))/9.0) ** 2 * 0.5;
		return error/len(testing_set[0]),error_rounded/len(testing_set[0])


class MnistNeuralNetwork(NeuralNetwork):
	@staticmethod
	def loadLabels(file,n_rows = 60000):
		with open(file, 'rb') as f:
			bytes= f.read(8)
			magic, size= struct.unpack(">II", bytes)

			labels_list= []
			for i in xrange(0,n_rows) :
				bytes= f.read(1)
				x = struct.unpack("B", bytes)
				labels_list.extend(x)
   
			labels= np.array(labels_list)

			labels.astype(float)
			
			labels= labels.T.reshape((n_rows,1))
		return labels
	
	@staticmethod
	def loadImages(file,n_inputs = 748,n_rows = 60000):
		with open(file, 'rb') as f:
			bytes= f.read(16)
			magic, size, rows, columns= struct.unpack(">IIII", bytes)
			images_list= []
			i=0
			while i<(n_rows):
				images_row = []
				for j in xrange(0,rows*columns) :
					bytes= f.read(1)
					if not bytes:
						break;
					x = struct.unpack("B", bytes)
					images_row.extend(x)
				if not bytes: break
				i+=1
				images_list.append(images_row)

		images= np.array(images_list)
		images= images.reshape((n_rows,784))
		return images

	@staticmethod
	def normalize(images,labels):
		for i in xrange(len(images)):
			for k in xrange(0,28):
				for d in xrange(0,28):
					images[i,k*28+d]= images[i,k*28+d]/255.0

		labels= np.true_divide(labels,9)
		return [images,labels]

	@staticmethod
	def displayImage(image):
		print "Image "
		for i in xrange(784):
			if image[i]>1:
				image[i]= 1
		for j in xrange(0,28):
			print '%03s' % image[j*28:j*28+28]

	@staticmethod
	def rotation(image):
		image= image.reshape(28,28)
		height, width = image.shape
		image = np.uint8(image)
		edges = cv2.Canny(image, 150, 200)
		for i in xrange(len(edges)):
			for j in xrange(len(edges[i])):
				if edges[i][j]>120:
					edges[i][j]=1
		lines = cv2.HoughLinesP(edges, 1, cv2.cv.CV_PI/180, 1, minLineLength=width / 2.0, maxLineGap=20)
		angle = 0.0
		if lines is None:
			return 0
		nlines = lines.size
			
		for x1, y1, x2, y2 in lines[0]:
			angle += np.arctan2(y2 - y1, x2 - x1)
		return angle * 10
	
	@staticmethod
	def deskew(image, angle):
		non_zero_pixels = cv2.findNonZero(image)
		center, wh, theta = cv2.minAreaRect(non_zero_pixels)
		root_mat = cv2.getRotationMatrix2D(center, angle, 1)
		rows, cols = image.shape
		rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
		return cv2.getRectSubPix(rotated, (cols, rows), center)

	@staticmethod
	def deskewSet(images):
		for i in xrange(len(images)):
			img= images[i]
			img= img.reshape(28,28)
			img = np.uint8(img)
			images[i] = MnistNeuralNetwork.deskew(img.copy(), MnistNeuralNetwork.rotation(img)).reshape(784)
		return images

def main():
	print "Loading training data"
	training_set_rows= 60000
	testing_set_rows= 10000

	np.random.seed(1)
	start = timeit.default_timer()
	h1= Layer(784,150)
	h2= Layer(150,30)
	output= Layer(30,1)

	labels= MnistNeuralNetwork.loadLabels("train-labels-idx1-ubyte",training_set_rows)
	images= MnistNeuralNetwork.loadImages('train-images-idx3-ubyte',784,training_set_rows)
	#MnistNeuralNetwork.displayImage(images[0])

	#Deskew training set
	images= MnistNeuralNetwork.deskewSet(images)

	network= MnistNeuralNetwork([h1,h2,output],MnistNeuralNetwork.normalize(images,labels))

	print "Training"
	network.run(500)

	print "Loading test data"
	test_images= MnistNeuralNetwork.loadImages('t10k-images-idx3-ubyte',784,testing_set_rows)
	#Deskew testing set 
	test_images= MnistNeuralNetwork.deskewSet(test_images)

	print "Testing"
	error, error_rounded = network.test(network.normalize(test_images,MnistNeuralNetwork.loadLabels('t10k-labels-idx1-ubyte',testing_set_rows)))	
	
	print 'Testing set error rate (%): ', error[0]*100
	print 'Testing set error rate with rounding (%): ', error_rounded*100

	stop = timeit.default_timer()
	print "Time: ", stop - start

main()

