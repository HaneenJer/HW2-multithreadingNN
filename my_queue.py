from multiprocessing import *

class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        
        self.lock = Lock()
        self.reader, self.writers = Pipe()
        return
        raise NotImplementedError("To be implemented")

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        
        #each writer can try to lock the pipe and write
        self.lock.acquire()
        try:
            self.writers.send(msg)
        finally:
            self.lock.release()
        return
        raise NotImplementedError("To be implemented")

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        #no need to lock, since the pipe lets one reader 
        return self.reader.recv()
        
        raise NotImplementedError("To be implemented")

