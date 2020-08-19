from main_with_runtime import main
import os
import subprocess

# should execute at each machine once 
def mainmain():
    write_dir = './write_dir/'
    num_ranks_in_server = 6
    num_ranks_prev_server = 0
    debug_mode = True

    module = 'models.vgg16.gpus=16_straight'
    config_path= 'models/vgg16/gpus=16_straight/mp_conf.json'
    batch_size = 8
    train_size = 640


    #args.data_dir = '/cmsdata/ssd0/cmslab/imagenet-data/raw-data/'
    synthetic_data = False
    distributed_backend = 'gloo'
    epochs = 3
    master_addr = '01.elsa.snuspl.snu.ac.kr'

    process_list = []
    file_list = []

    for i in range(num_ranks_in_server):
        rank = i + num_ranks_prev_server
        local_rank = i
        if debug_mode:
            p = subprocess.Popen(["python", "main_with_runtime.py", 
                "--module", module, "--config_path", config_path, 
                "--num_ranks_in_server", str(num_ranks_in_server),
                "--rank", str(rank), "--local_rank", str(local_rank), 
                "--b", str(batch_size), '--epochs', str(epochs), '--train_size', str(train_size)], 
                stdout = None, stderr=subprocess.STDOUT, text=True)
        else:
            p = subprocess.Popen(["python", "main_with_runtime.py", 
                "--module", module, "--config_path", config_path, 
                "--num_ranks_in_server", str(num_ranks_in_server),
                "--rank", str(rank), "--local_rank", str(local_rank), 
                "--b", str(batch_size), '--epochs', str(epochs), '--train_size', str(train_size)], 
                stdout = subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        process_list.append(p)

        filename = write_dir + "gpu16" + "_" + str(rank) + ".txt"
        file_list.append(filename)
    try:
        for p in process_list:
            p.wait()
        
        for i in range(len(process_list)):
            if os.path.exists(file_list[i]):
                f = open(file_list[i], "w")
            else:
                f =  open(file_list[i], "x")

            if debug_mode == False:
                f.write(process_list[i].communicate()[0])

    except: 
        for p in process_list:
            p.kill()

if __name__ == '__main__':
    mainmain()
