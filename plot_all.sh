#!/bin/bash
python plot.py -x=env_steps -y=avg_return -e=HalfCheetah -b=5000 &
python plot.py -x=env_steps -y=log_probs_alpha -e=HalfCheetah -b=5000 &
python plot.py -x=env_steps -y=avg_return -e=Hopper -b=15000 &
python plot.py -x=env_steps -y=log_probs_alpha -e=Hopper -b=5000 &
wait