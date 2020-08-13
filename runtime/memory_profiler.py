import torch.cuda
import psutil
import matplotlib.pyplot as plt 

class MemProfiler:
    def __init__(self):
        self.records = [] # dict list
        self.periods = [] # string list
        
    def call_profiler(self, period):
        
        gpu_mem_mb = torch.cuda.memory_allocated() >> 20
        cpu_mem_mb = psutil.virtual_memory().used >> 20
        mem_dict = {'gpu': gpu_mem_mb, 'cpu':cpu_mem_mb}
        self.periods.append(period)
        self.records.append(mem_dict)
        '''
        print("---------------------------------- %s --------------------------" % period)
        print("Current GPU memory: %d, CPU memory: %d" % (gpu_mem_mb, cpu_mem_mb))
        if len(self.records) > 1:
            print("Delta GPU memory: %d, CPU memory: %d" % (gpu_mem_mb - self.records[-2]['gpu'], 
                cpu_mem_mb - self.records[-2]['cpu']))
        '''

    def draw_graph(self, filename):
        return
        x = [i for i in range(len(self.records))]
        y = [i['gpu'] for i in self.records]


        #f = open('raw.txt', 'w')
        #f.write('CPU: ' + str([i['cpu'] for i in self.records])+"\n")
        #f.write('GPU: ' + str([i['gpu'] for i in self.records])+"\n")
        print("y: ", y)
        #plt.plot(x, y)
        
        #plt.savefig(filename)
        print("success drawing graph")


