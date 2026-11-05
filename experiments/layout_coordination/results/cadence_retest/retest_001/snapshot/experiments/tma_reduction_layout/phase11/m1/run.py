"""CPU-only audit entrypoint. No Modal, CUDA driver, compiler or GPU imports."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ['CUDA_VISIBLE_DEVICES']=''
from experiments.tma_reduction_layout.phase11.m1 import profile,pipeline,mechanisms,plot,report


def main():
    profile.run();pipeline.run();mechanisms.run();plot.run();report.run()


if __name__=='__main__':main()
