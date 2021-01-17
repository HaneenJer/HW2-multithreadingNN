import multiprocessing
import scipy.ndimage as sn
import numpy as np
from math import *
import random
Len = 784
Dim = (28,28)
root = 28 

class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result
        
        #training images
        self.data_images = training_data[0]
        #labels images
        self.label_images = training_data[1]
        
        self.batch_size = batch_size
        return
        raise NotImplementedError("To be implemented")

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
		#matrix of size 28X28
        new_img = image.reshape(Dim)
		#rotate the image 
        new_img = sn.rotate(new_img, angle, reshape=False)
        return new_img.reshape((Len,))
         
        raise NotImplementedError("To be implemented")

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
		#reshape to get matrix 28X28 
        shift_img = np.reshape(image, Dim)
		
		#filling with 0		
        for j,i in zip(range(dx),range(root)):
            shift_img[i][j] = 0
        for i,j in zip(range(dy),range(root)):
            shift_img[i][j] = 0
        
        #shifting the image  
        shift_img = np.roll(shift_img, -dy, axis=0)
        shift_img = np.roll(shift_img, -dx, axis=1)
		#after done with shifting the image, back to original shape
        shift_img = np.reshape(shift_img, Len)
        return shift_img
        raise NotImplementedError("To be implemented")

    @staticmethod
    def step_func(image, steps):
        '''Transform the image pixels acording to the step function

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        steps : int
            The number of steps between 0 and 1

        Return
        ------
        An numpy array of same shape
        '''
		
        assert steps > 1
        stps = (1 / (steps - 1)) 
		#for each pixel in image, we change it accordingly 
        for i in range(Len):
            image[i] = stps * floor(steps * image[i])
        return image
        raise NotImplementedError("To be implemented")

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        img_skew = image.reshape(Dim)
        final_skew = img_skew
        for x in range(root):
            for y in range(root):
                tmp = floor(x+y*tilt)
                #if it's out of bounds then black - aka 0
                if( tmp > 27 or tmp < 0 ):
                    res = 0
				#otherwise we calculate skew	
                else:
                    res = img_skew[y][tmp] 
               
                final_skew[y][x] = res
        final_skew = np.reshape(final_skew, Len) 
        return final_skew
        raise NotImplementedError("To be implemented")

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
		#choosing random values to use later as parameters for the functions
        angle = np.random.randint(-9,9)
        dx = np.random.randint(-1,2)
        dy = np.random.randint(-1,2)
        steps = np.random.randint(2,9)
        tilt = np.random.uniform(-0.1, 0.1)
        #running the functions
        return (self.step_func(self.skew(self.rotate(self.shift(image,dx,dy),angle),tilt),steps))

        raise NotImplementedError("To be implemented")

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
		
            job = self.jobs.get()
        #no jobs left to do 
            if job is None:  
                self.jobs.task_done()
                break
                
            sample_len = self.batch_size
            sample_seq = range(self.data_images.shape[0])
			# choosing a random data sample sized batch_size 
            idx = random.sample(sample_seq, sample_len)
			#taking the images from the data sample 
            data = self.data_images[idx]
			#taking the labels from the data sample
            label = self.label_images[idx]
            
            batch = (data, label)
            batch_len = len(batch[0])
            #processing images in order to save it later in result queue
            for i in range(batch_len):
                batch[0][i] = self.process_image(batch[0][i])  

            self.result.put(batch)  
            
            self.jobs.task_done()
            
        return 
        raise NotImplementedError("To be implemented")
   