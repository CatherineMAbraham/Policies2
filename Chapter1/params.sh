#!/bin/bash

> params.csv

# First three pairs with both fouractions and euler
pos_ori_pairs_both=("0.001 1" "0.005 3" "0.008 5")
for pair in "${pos_ori_pairs_both[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  for act in fouractions euler; do
      echo "$pos,$ori,$act" >> params.csv
  done
done

# Last two pairs with only euler
pos_ori_pairs_euler=("0.001 5" "0.008 1")
for pair in "${pos_ori_pairs_euler[@]}"; do
  pos=$(echo $pair | cut -d' ' -f1)
  ori=$(echo $pair | cut -d' ' -f2)
  echo "$pos,$ori,euler" >> params.csv
done

