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

* Move model files to the model_served directory.
* You can expose the server of the model by running

```
python runner_serve.py
```

* Then you can send requests as follows

```
curl -H "Content-Type: application/json" -X POST -d '{"text":"on april first i need a flight going from phoenix to san diego"}' 0.0.0.0:5000/classify
```