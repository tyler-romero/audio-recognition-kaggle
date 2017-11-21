# audio-recognition-kaggle

Competition website can be found [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/)

## Notes

* Can't take dependencies on anything other than Tensorflow 1.4 (and its dependencies (ie numpy))
* Model must be below 5,000,000 bytes
* Model must be runable as a frozen GraphDef
* Model must run in less than 200ms on a Raspberry Pi 3 without overclocking

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

* Go to [comet.ml](https://www.comet.ml/tyler-romero) to view run metrics

## The benchmark

* Meant to be run on a raspberry pi 3
* Used for the actual competition
* The benchmarking binary takes a pretrained tensorflow graph

```bash
./benchmark.sh
```

## References

* [Small-Footprint Keyword Spotting Using DNNs](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42537.pdf)
* [CNNs for Small-footprint Keyword Spotting](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)
* [Compressing DNNs using a Rank-Constrained Topology](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf)
