#!/bin/bash

# generate preprocessed GT
python main_replica.py 0 test all > logs/main_replica_gt_test.log & 
python main_replica.py 0 valid all > logs/main_replica_gt_val.log & 
python main_replica.py 0 train1 all > logs/main_replica_gt_train1.log 
python main_replica.py 0 train2 all > logs/main_replica_gt_train2.log 

# generate preprocessed Experts
python main_replica.py 1 test all > logs/main_replica_gt_test.log & 
python main_replica.py 1 valid all > logs/main_replica_gt_val.log & 
python main_replica.py 1 train1 all > logs/main_replica_gt_train1.log & 
python main_replica.py 1 train2 all > logs/main_replica_gt_train2.log & 

