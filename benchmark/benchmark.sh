#!/usr/bin/env bash

# I think this has to run on a raspberry pi with rasbian

echo Downloading data
curl -O https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip
unzip speech_commands_v0.01.zip

echo Downloading benchmarking executable
curl -O https://storage.googleapis.com/download.tensorflow.org/deps/pi/2017_10_07/benchmark_model
chmod +x benchmark_model

echo Benchmarking model
./benchmark_model \
--graph=conv_actions_frozen.pb \
--input_layer="decoded_sample_data:0,decoded_sample_data:1" \
--input_layer_shape="16000,1:" \
--input_layer_type="float,int32" \
--input_layer_values=":16000" \
--output_layer="labels_softmax:0" \
--show_run_order=false \
--show_time=false \
--show_memory=false \
--show_summary=true \
--show_flops=true

