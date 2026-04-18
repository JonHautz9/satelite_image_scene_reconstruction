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