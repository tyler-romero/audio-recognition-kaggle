# audio-recognition-kaggle
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/

## Setup

* Download the training and testing data and place in the data folder
* Download the benchmarking script

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

## The benchmark

* Meant to be run on a raspberry pi 3
* Used for the actual competition
* The benchmarking binary takes a pretrained tensorflow graph

```bash
./benchmark.sh
```
