#!/bin/bash

python parentingGLMHMM.py hydra.run.dir=. num_states=1 random_state=999
python parentingGLMHMM.py hydra.run.dir=. num_states=2 random_state=999
python parentingGLMHMM.py hydra.run.dir=. num_states=3 random_state=999
python parentingGLMHMM.py hydra.run.dir=. num_states=4 random_state=999
