# Fall-detection（摔倒/跌倒检测）
Fall-detection（摔倒/跌倒检测）in the room

 this fall-detection is based on [darknet](https://pjreddie.com/darknet/yolo/).


## Examples
   we first detections humans in the room, then we use some simple ways to judge whether he or she is falling down.
![fall detection example](https://github.com/qiaoguan/Fall-detection/blob/master/demo.gif)

### fall-detection algorithms

  * detection human
  * classify whether fall down or not(still on the way)/custom rules  
## Installation

### Requirements

  * Python and opencv
  * Linux (Windows and Mac os are not officially supported, but should work)

### Installation Options:

#### Install on Linux

First, make sure you have install python and opencv environment


Then, install this module :

```bash
git clone https://github.com/qiaoguan/Fall-detection
cd Fall-detection
make
```

If you are having trouble with installation, you can Issue me!

### run the demo

  First, download [yolo.weights](https://pan.baidu.com/s/1eTqopgQ), the password is： bp6c.
  Then, install this module :

  ```bash
  python gq.py
  ```

## Thanks

* Many, many thanks to [pjreddie](https://pjreddie.com/darknet/yolo/) for his Great work!
  
