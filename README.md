# Video Object Detection 

## Install

````
$ pip install -r requirements.txt
````

Additionally requires:
- OpenCV 3.4+
- Movidius NCSSDK v2
- Tensorflow (optional)

## Setup

1. Install opencv. 
2. Add models to `./models/<model_dir>`
3. Add `./CONFIG` file; see `./CONFIG_EXAMPLE` for reference. 


## Usage

#### Monitor Mode

#### Inference Mode

````
$ python -m src.video -i 0 -c CONFIG
````

#### Test Mode

````
$ python -m src.video -i 0 -o tmp/test.mov -c CONFIG
````
