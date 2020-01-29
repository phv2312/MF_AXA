# MF_AXA

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phv2312/MF_AXA.git
```

## Usage

```python
import cv2
from mf_axa.post_process.split_coord_founder import SplitCoordFounder

# load image
im_fn = "where is your MULTIPLE_DATA image path"
im = cv2.imread(im_fn)

# load model
founder = SplitCoordFounder()

# running
split_coord, area, direction = founder.process(, direction=None) # you can specify the direction ['h','w'] to get more accurate result

if direction == 'w':
    col = split_coord
    if col > -1:
        cv2.line(im, (col, 0), (col, im.shape[0]), (255, 0, 0), thickness=10)
    else:
        print("Cannot split")
elif direction == 'h':
    row = split_coord
    if row > -1:
        cv2.line(im, (0, row), (im.shape[1], row), (255, 0, 0), thickness=10)
    else:
        print("Cannot split")

imgshow(im)
```
