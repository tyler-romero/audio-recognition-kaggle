#!/usr/bin/env bash

rm data/train/audio/silence -rf
rm data/train/audio/unknown -rf
python code/prepare_data.py