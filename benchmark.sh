#!/usr/bin/env bash

# Replace the .pb file with the pretrained tensorflow graph

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