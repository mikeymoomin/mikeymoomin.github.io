---
layout: post
title: "It's Easay to get Started"
date: 2025-03-17 08:00:00
categories: learning
---

# It's Easy to Get Started! 

## Getting the Dataset

Lesson 1 shows you how to great a model that classifies images - the lesson talks the viewer through forming a 'Is it a bird?' identifier model. There were a few mind = blown moments in this lecture the first was the **Duck-Duck-Go image search API**. Here, the API enabled me to quickly and efficiently put together a small dataset of images taken directly from a web search on the Duck-Duck-Go (DDG) browser. This was perforemd in the following code:

```python
# Install/import dependancies and define functions 
!pip install -Uqq fastai 'duckduckgo_search>=6.2'
from duckduckgo_search import DDGS
from fastcore.all import *
def search_images(keywords, max_images=200): return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')
import time, json

from fastdownload import download_url
from fastai.vision.all import *

# Tidy for-loop
searches = 'forest','bird'
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    time.sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)
```

Here, we can see a few pretty neat things, the importing of the DDG search functionality and then the definition of that search function which seeks to take in the keyword (i.e. the search) and return a maximum ot 200 images per search. You can then see a very tidy for-loop that iterates over our searchs of 'bird' and 'forest' to both create a seperate file path for both and download up to 200 images for each before resizing. 

It's worth noting here that not all images will have successfully downloaded and therefore we should remove them from the dataset with the following code:

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
## Training the Model

So, we now have a dataset of up to 200 images of birds, and up to 200 images of forest and general woodland. We now need to train the model. We can do this with DataLoaders, specifically the following code:

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)
```

There are a few things going on in that chunck of code, and it's important to understand each bit and its purpose. 

```python
blocks=(ImageBlock, CategoryBlock),
```
First, the above code tells the DataLoader that our input is an image and out output will be a category - in this case it'll be either a 'Forest' or a 'Bird';

```python
get_items=get_image_files,
```
Then, the above code effectively hands the DataLoader the images to use for testing and training by returning a list of all image file paths;

```python
splitter=RandomSplitter(valid_pct=0.2, seed=42),
```

Next, we are defining the split of the data between the training and testing set randomly with 20% of the input impages being used as testing data;

```python
get_y=parent_label,
```

We are then labelling the images with the name of the parent file that the image is situated in - in this case 'Bird' or 'Forest';

```python
item_tfms=[Resize(192, method='squish')]
```

Finally, we are resizing all images to 192x192 pixels by squishing as opposed to cropping so that we don't lose any of the images.

After all of that, we can now traing the model - and here comes the next thing that blew my mind. There are easily downloadable models available for free that are already great, and just need some fine tuning for your use case. 

Here, we use the resnet18 model. This is a well known convolutional neural network (CNN) with 18 layers, used mainly for image classification tasks. It's a popular model because it achieves good accuracy with relatively low computational cost, making it excellent for quick experimentation and baseline models.

With the following code therefore,  we can initialise a CNN-based learner designed specifically for vision tasks such as image classification for our use case. This is done easily by using the fast.ai function 'vision_learner' which initially asks for our DataLoader (dls) in order to pass through our bird and forest dataset, then parsing the model resnet18 before defining our metric to measure success - in this case error_rate. After that, we then finetune that CNN with our data using the fine_tune method which in this case over the course of 3 epochs. This works by updating the models weights and biases by calculating the loss and using backpropagation - This was another lightbulb moment that made [Andrew Ng and STATQuests's](https://mikeymoomin.github.io/2025/03/17/first-cs-learning-post/) lessons start to make sense!

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

The output is as below. This shows the three training runs, with a error rate declining right to 0 in a very short amount of time - perfect!

| epoch | train_loss | valid_loss | error_rate | time  |
|-------|------------|------------|------------|------|
| 0     | 0.018716   | 0.016954   | 0.016129   | 00:01 |
| 1     | 0.009592   | 0.003295   | 0.000000   | 00:01 |
| 2     | 0.010859   | 0.001002   | 0.000000   | 00:01 |

We note have a CNN model that can very accurately predict whether a bird is in an image!

