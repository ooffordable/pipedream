# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

r"""
`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.
The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.
In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.
**How to use this module:**
1. Single-Node multi-process distributed training
::
    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)
2. Multi-Node multi-process distributed training: (e.g. two nodes)
Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*
::
    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)
Node 2:
::
    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)
3. To look up what optional arguments this module offers:
::
    >>> python -m torch.distributed.launch --help
**Important Notices:**
1. This utilty and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.
2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:
Parsing the local_rank argument
::
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()
Set your device to local rank using either
::
    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs
or
::
    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run
3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.
::
    torch.distributed.init_process_group(backend='YOUR BACKEND',
                                         init_method='env://')
4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.
::
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)
Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility
5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.
.. warning::
    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.
"""


import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--dist_world_size", type=int, default=6)
    parser.add_argument("--nproc_this_node", type=int, default=6)
    parser.add_argument("--num_prev_ranks", type=int, default=0)

    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--module_name", type=str, default=None,
                        help="just for name of log file")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.dist_world_size

    # set PyTorch distributed related environmental variables
    processes = []
    file_list = []
    write_dir = "./write_dir/"
    
    for local_rank in range(0, args.nproc_this_node):
        # each process's rank
        dist_rank = args.num_prev_ranks + local_rank

        # spawn the processes
        cmd = [sys.executable,
               "-u",
               args.training_script,
               "--rank={}".format(dist_rank),
               "--local_rank={}".format(local_rank)] + args.training_script_args

        #process = subprocess.Popen(cmd, stdout = subprocess.PIPE, text=True)
        process = subprocess.Popen(cmd)
        processes.append(process)
        args.module_name = "gpu8"
        filename = write_dir + args.module_name + "_" + str(dist_rank) + ".txt"
        file_list.append(filename)
        print(filename)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

    for i in range(len(processes)):
        if os.path.exists(file_list[i]):
            f = open(file_list[i], "w")
        else:
            f =  open(file_list[i], "x")
        f.write(processes[i].communicate()[0])

if __name__ == "__main__":
    main()