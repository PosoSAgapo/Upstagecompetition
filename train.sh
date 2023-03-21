#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N GRU4RecF_seq_rec
#PJM -o GRU4RecF_seq_rec
#PJM -e GRU4RecF_seq_rec

source /work/gk77/k77025/.zshrc
python train.py --model GRU4RecF
python seq_inference.py --model GRU4RecF
