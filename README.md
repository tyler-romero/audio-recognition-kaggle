# audio-recognition-kaggle

Competition website can be found [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/)

## Notes

* Can't take dependencies on anything other than Tensorflow 1.4 (and its dependencies (ie numpy))
* Model must be below 5,000,000 bytes
* Model must be runable as a frozen GraphDef
* Model must run in less than 200ms on a Raspberry Pi 3 without overclocking

## Setup

* Download the training and testing data and place in a `data` folder
* Download the benchmarking script and place in a `benchmark` folder

```bash
curl -O https://storage.googleapis.com/download.tensorflow.org/deps/pi/2017_10_07/benchmark_model
chmod +x benchmark_model
```

## Training

* Generates a tensorflow graph
* Spits out metrics occasionally
* Run the following script

```bash
./train.sh
```

This is just for convenience, internally it is just doing something like this:

```bash
python code/train.py --model_architecture SimpleConv
```

Go to [comet.ml](https://www.comet.ml/tyler-romero) to view run metrics

## The benchmark

* Meant to be run on a raspberry pi 3
* Used for the actual competition
* The benchmarking binary takes a pretrained tensorflow graph

```bash
./benchmark.sh
```

Internally, it is doing this:

```bash
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
```

This is what will be run in order to actually evaluate our model on a raspberry pi, so we need to make this work.

## References

* [Small-Footprint Keyword Spotting Using DNNs](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42537.pdf)
* [CNNs for Small-footprint Keyword Spotting](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
* [Compressing DNNs using a Rank-Constrained Topology](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf)
