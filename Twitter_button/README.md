# Twitter following button detection

This is a simple challenge to detect the Twitter following buttons, and the outputs are the bounding boxes for the button detection and figure visulization with color masks on the buttons.

The output looks like this:

![Twitter following button](https://github.com/jjz369/Mask_RCNN/blob/master/Twitter_button/images/5_detected.png)

The training and validation datasets are based on 4 twitter following page screenshots similar to the above figure (as shown 1.png ~ 4.png in image folder). Randomly cut the figures into smaller ~50 images, and separated into training and validation datasets (as shown in datasets/train and datasets/val). The annotations (json format) are obtained using [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/).  


## Detect the following buttons

To detect the Twitter following buttons, use the following command:

```bash
python3 button.py detect --weights=/path/to/mask_rcnn/mask_rcnn_button.h5 --image=<file name or URL>
```

## Run Jupyter notebooks
Open the `Twitter_button_detection.ipynb` Jupyter notebook. This notebook shows the detailed flow of the detection.
 
## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 button.py train --dataset=/path/to/button/datasets/ --weights=coco
```

Resume training a model that you had trained earlier
```
python3 button.py train --dataset=/path/to/button/datasets/ --weights=last
```

Train a new model starting from ImageNet weights
```
python3 button.py train --dataset=/path/to/button/datasets/ --weights=imagenet
```

The training process use a configuration for only one epoch in heads training and one epoch in all weights training for simplicity. 
