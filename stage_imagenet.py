"""Stage ImageNet data onto local SSDs."""

import argparse
import os
import os.path
import subprocess
import time

parser = argparse.ArgumentParser(
    description='Stage ImageNet to SSDs.')
parser.add_argument(
    'basedir', type=str,
    help='Base directory containing {train,val}.tar')


def get_num_nodes():
    """Return the number of nodes in the job."""
    # TODO: Support Slurm, etc.
    if 'NUM_COMPUTE_NODES' in os.environ:
        return int(os.environ['NUM_COMPUTE_NODES'])
    raise RuntimeError('Could not get world size')


def stage_file(basedir, filename, dstdir='imagenet'):
    """Stage a single file."""
    ssd = os.environ['BBPATH']
    # Broadcast tar file.
    subprocess.run(
        ['jsrun',
         '--nrs', str(get_num_nodes()),
         '--rs_per_host', '1',
         '--tasks_per_rs', '4',
         '--cpu_per_rs', 'ALL_CPUS',
         'dbcast',
         os.path.join(basedir, filename),
         os.path.join(ssd, dstdir, filename)],
        check=True)
    # Untar.
    subprocess.run(
        ['jsrun',
         '--nrs', str(get_num_nodes()),
         '--rs_per_host', '1',
         '--tasks_per_rs', '1',
         '--cpu_per_rs', 'ALL_CPUS',
         'tar', 'xf',
         os.path.join(ssd, dstdir, filename),
         '-C', os.path.join(ssd, dstdir)],
        check=True)


if __name__ == '__main__':
    args = parser.parse_args()
    start = time.perf_counter()
    stage_file(args.basedir, 'train.tar')
    stage_file(args.basedir, 'val.tar')
    total_time = time.perf_counter() - start
    print(f'Staged in {total_time:.4f} s')
