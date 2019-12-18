# Description
This project trains an ML model for joint intent detection and slot filling using the ATIS dataset.

# Architecture
The architecture chosen in this implementation is similar to the “Attention-based RNN model for joint intent detection and slot filling” in the following paper. [paper](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/1352.PDF)
The only difference is that the context vector was not concatenated with the time-step hidden state vector in slot filling. One reason for choosing this model versus the encoder-decoder model is that it is computationally more efficient since it only reads the input once – as opposed to twice in the encoder-decoder model.
Pre-trained word embeddings have been used in this task. Some reserved tokens such as BOS, EOS and unknown words have been initialised randomly.

# Setting up
* Create an environment - e.g. using conda - and install the requirements.
```
pip install -r requirements.txt
```

* Download the google word vectors using the get_resources.sh script

```
sh get_resources.sh
```


# Training

The best training settings are set to default. You can change training settings by using different parameters.
Run the training with the default parameters using

```
python runner_train.py  ./data/models/name_of_model/
```


# Testing

You can test the model to create a results.csv file by running. Note that running this will overwrite any already existing results.csv file


```
python runner_test.py  ./data/models/name_of_model/
```

# Serving

* The directory for the served model is data/models/best/
* Move any model you want to serve to there.
* You can expose the server of the model by running

```
python runner_serve.py
```

* Then you can send requests as follows

```
curl -H "Content-Type: application/json" -X POST -d '{"text":"on april first i need a flight going from phoenix to san diego"}' 0.0.0.0:5000/classify
```