#!/bin/bash

# python ./data/transform_reward.py --b1 10.0 --c1 2.0 --b2 1.0 --c2 5.0 --c3 4.0 --eta 0.2 --trigger_prob 0.005 ; python training.py ; python testing.py


python ./data/transform_reward.py --b1 5.0 --c1 5.0 --b2 5.0 --c2 5.0 --c3 5.0 --eta 0.2 --trigger_prob 0.005 ; python training_v2.py ; python testing_v2.py

python ./data/transform_reward.py --b1 5.0 --c1 15.0 --b2 5.0 --c2 5.0 --c3 5.0 --eta 0.2 --trigger_prob 0.005 ; python training_v2.py ; python testing_v2.py
