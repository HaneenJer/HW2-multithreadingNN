import os
from network import *
from my_queue import *
import multiprocessing 
from preprocessor import *

class IPNeuralNetwork(NeuralNetwork):
    result = MyQueue()
    jobs = multiprocessing.JoinableQueue()
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        #get the number of cpus
        workers_num = int(os.environ['SLURM_CPUS_PER_TASK'])
        
        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        workers = []
        for i in range(workers_num):
            workers.append(Worker(self.jobs, self.result, training_data,self.mini_batch_size))
        
		# 2. Set jobs
        for worker in workers:
            worker.start()
		
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        
        # 3. Stop Workers
        for worker in workers:
            self.jobs.put(None)
            
        self.jobs.join()
        return
        raise NotImplementedError("To be implemented")
        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        batches_num = self.number_of_batches
        
        
        for i in range(batches_num):
            self.jobs.put("sample")
        
        #queue to save the augmented images from workers
        res = []
        
        #for each batch get the augmented image and add it to the res queue
        for i in range(batches_num):
            res.append(self.result.get())
           
        return res
        
        raise NotImplementedError("To be implemented")

    
