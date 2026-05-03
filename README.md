# Satelite Image Scene Reconstruction

## Environment Setup
### You will need to use an environment with Python < 3.12
```
conda create -n cs445-final-project python=3.11
pip install -r requirements.txt
```

### Use this if pip install fails to because it can't find a Cython module
```
set PYTHONUTF8=1 # For UnicodeDecodeError on Windows
pip install intelligent-scissors --no-binary :all: --no-build-isolation
```

### Install Forked Version of Intelligent Scissors
#### How to Build the wheel
##### Install Build Tools
Install `wheel` and `setuptools` in your environment. 
```
install wheel setuptools
```
##### Build the Wheel 
Run the following command in the root project directory to create the `dist/` folder containing the `.whl` file.
```
python setup.py bdist_wheel
```
##### Alternative using build 
`pip install build` then `python -m build`

#### How to Install the Wheel
Once the `.whl` file is generated install using pip
```
pip install dist/intelligent_scissors-0.1.0-py3-none-any.whl
```

## Known Issues Running the Code
### Intelligent Scissors Generic Preprocess Image Error
There is an issue with intelligent-scissors reading images that are screenshotted from Google Earth. You may run into a generic non descriptive runtime error in `scissors\utils.py", line 72, in preprocess_image`. To get around this you can run the `rgb_save()` method in test_intelligent_scissors.py to convert to an RGB image that can be read by intelligent-scissors.
