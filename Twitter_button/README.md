# Twitter following button detection

This is a simple challenge to detect the Twitter following buttons, and then we will output the bounding boxes and visualize the page with color masks on the buttons.

The output looks like this:

![Twitter following button](https://github.com/jjz369/Mask_RCNN/blob/master/Twitter_button/images/5_detected.png)

The training and validation datasets are based on 4 twitter following page screenshots similar to the above figure. Randomly cut the figures into smaller ~50 images, and separated into training and validation datasets. The annotations (json format) are obtained using [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/).  


## Detect the following buttons

To detect the Twitter following buttons, use the following command:

```bash
python3 button.py detect --weights=last --image=./datasets/test/5.jpg
```

## Run Jupyter notebooks
Open the `Twitter_button_detection.ipynb` Jupyter notebook. This notebook shows the detailed flow of the detection.
 
## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 button.py train --dataset=./datasets/train --weights=coco
```

Resume training a model that you had trained earlier
```
python3 button.py train --dataset=./datasets/train --weights=last
```

Train a new model starting from ImageNet weights
```
python3 button.py train --dataset=./datasets/train --weights=imagenet
```

The training process use a configuration for only one epoch in heads training and one epoch in all weights training for simplicity. 
