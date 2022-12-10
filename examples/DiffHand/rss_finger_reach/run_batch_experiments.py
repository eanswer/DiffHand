import os
from multiprocessing import Process, Queue
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ours')
parser.add_argument('--num-seeds', type=int, default=5)
parser.add_argument('--num-processes', type=int, default=1)
parser.add_argument('--save-dir', type=str, default='./results/2021-02-20/')

args = parser.parse_args()

''' generate seeds '''
seeds = []
for i in range(args.num_seeds):
    seeds.append(i * 10)

''' generate commands '''
commands = []
for i in range(args.num_seeds):
    seed = seeds[i]
    save_dir = os.path.join(args.save_dir, args.method, str(seed))
    if args.method == 'ours':
        cmd = 'python example_finger_reach.py '\
            '--seed {} '\
            '--save-dir {} '\
            '--visualize False'\
                .format(seed, save_dir)
    elif args.method == 'control-only':
        cmd = 'python example_finger_reach.py '\
            '--seed {} '\
            '--save-dir {} '\
            '--no-design-optim '\
            '--visualize False'\
                .format(seed, save_dir)
    elif args.method == 'CMA':
        cmd = 'python grad_free.py '\
            '--seed {} '\
            '--save-dir {} '\
            '--optim CMA '\
            '--max-iters 10000 '\
            '--popsize 10 '\
            '--visualize False'\
                .format(seed, save_dir)
    elif args.method == 'OnePlusOne':
        cmd = 'python grad_free.py '\
            '--seed {} '\
            '--save-dir {} '\
            '--optim OnePlusOne '\
            '--max-iters 10000 '\
            '--visualize False'\
                .format(seed, save_dir)
    commands.append(cmd)

def worker(input, output):
    for cmd in iter(input.get, 'STOP'):
        ret_code = os.system(cmd)
        if ret_code != 0:
            output.put('killed')
            break
    output.put('done')
    
# Create queues
task_queue = Queue()
done_queue = Queue()

# Submit tasks
for cmd in commands:
    task_queue.put(cmd)

# Submit stop signals
for i in range(args.num_processes):
    task_queue.put('STOP')

# Start worker processes
for i in range(args.num_processes):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results
for i in range(args.num_processes):
    print(f'Process {i}', done_queue.get())