# Dangsan Object Detection Baseline
## Installation

* Installed on ~~`Python 3.9.7`~~ `Python 3.7.9`
```bash
$ git clone https://github.com/yeongseon/dangsan-object-detection.git
$ cd dangsan-object-detection
$ pip install -r requirements.txt
```
## Run

1. Create a config file in ```config/``` by following templates
2. Launch the script ```main.py``` with the config file : 
``` 
$ python main.py --config_file config/config.yaml
```
3. Also, you can change use options to change launching configuration :
``` 
$ python main.py --config_file config/config.yaml --batch_size 2 --num_workers 2 --epoch 20 --csv_file path --root_path path
```
And you can use the ```--fast_dev_run``` option

The config file is read by the ```Trainer``` in ```agents/trainer.py``` which launch the training.
