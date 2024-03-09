# Conventional Image Processing Techniques
Apply conventional image processing technique to localize objects.

### Installation

```
conda create -n arkeus python==3.10
conda activate arkeus
pip install opencv-python
```


### Run solution

```
python main.py
```

After running the above `main.py` script, the result image is exported into `results` folder.

NOTE: To get upper and lower HSV thresholds, run `hsv_calibration.py` to get desired values.

Then, modify `lower-hsv` and `upper-hsv` arguments in `main.py` script with corresponding values to extract better.


