---
layout: post
title: "Tom Cruise and Data Cleaning"
date: 2025-03-18 08:00:00
categories: learning
---

## The idea

A Deepfake a video of a person in which their face or body has been digitally altered so that they appear to be someone else. Auditors and Finance professionals are increasingly warey of the threat of fraud on account of this technology. Indeed, British engineering firm Arup were recently [scamed out of $25 million](https://edition.cnn.com/2024/05/16/tech/arup-deepfake-scam-loss-hong-kong-intl-hnk/index.html) after a finance worker sent a payment to a scammer after having gained acceptance from who he thought was the CFO and other members of staff - all of whom turned out to be deepfake re-creations.

Furthermore, the risk of Deepfakes infultrating the politics in the West opens the door for rampant misinformation and for bad actors to to act maliciously. The question is raised as to how we can combat this new technology - I put to you that Neural Networks may have a role to play in this.

As part of lesson 2, which focuses on manipulation of data and deploying a model to Hugging Face; I wanted to see if an image classification Convilutional Neural Network (CNN) could identify the difference between a deekfake and a real image of someone. For my use case, who better to pick that Tom Cruise! The man has had his face blased through our televisions for decades, and so there are plenty of real images of him. Interestingly however, many people showcasing deepfake videos seem to be [using him as their use case](https://youtube.com/shorts/oPbuyJqSQ2k?si=9V-J_82hiu7ZyJ3h), and so there are also plenty of deepfakes of him on the internet. The best bit, we can train a model to showcase this so quickly - I feel the need—the need for speed!

## Setting Up the Model

Setting up the model takes a very similar approach to the 'Is it a Bird?' model as prepared in Lesson 1. We will start by downloading the dependancies, defining a function to return a maximum of 200 images, removing the failed downloads, creating files and searching for 'Tom Cruise' and 'Deepfake Tom Cruise' and then defining our DataLoader. All of this has been done in the below code - it's worth noting that this process is performed in a Jupyter Notebook, it makes the process so much easier and less stop-start!

```python
pip install -Uqq fastai 'duckduckgo_search>=6.2'
from duckduckgo_search import DDGS  
from fastcore.all import *
import time, json
from fastdownload import download_url
from fastai.vision.all import *

def search_images(keywords, max_images=200): return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

searches = 'Deepfake Tom Cruise photo','Tom Cruise photo'
path = Path('tom_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    time.sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)
```

Now that we've done all of that, we can train the ResNet18 CNN on our Tom Cruise image data, and hopefully get a decent output. We can do that with the following code:

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
And out output is as follows:

| Epoch | Train Loss | Valid Loss | Error Rate | Time  |
|-------|------------|------------|------------|--------|
| 0     | 1.061800  | 0.620973  | 0.263889  | 00:42  |
| 0     | 0.442230  | 0.562974  | 0.208333  | 00:57  |
| 1     | 0.342766  | 0.414894  | 0.125000  | 00:58  |
| 2     | 0.251430  | 0.416044  | 0.138889  | 00:56  |

So from the above, we can see that I've trained the model initially over 3 epochs and the model is getting gradually better with the error rate reducing from 26.4% to 13.9% and then flatting out around there. This is our benchmark, we can now clean the data and hopefully get some imporvements.

## Cleaning the Data

The first thing we can do is look at the pictures that the model struggled with the most in both the training set and the testing set and thanks to the Fast.ai library we can reallocate or delete imagesed in the DataLoader that aren't correct or correctly classified. We can do this with the following code:

```python
# First run this
from fastai.vision.widgets import *
cleaner = ImageClassifierCleaner(learn)
cleaner

# Then when you've done do this
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

I will then have to re-define the DataLoader before finetuning again - This time I chose to fine tune over 4 epochs with the following results:

| Epoch | Train Loss | Valid Loss | Error Rate | Time  |
|-------|------------|------------|------------|--------|
| 0     | 0.937698  | 0.556013  | 0.228571  | 00:38  |
| 0     | 0.416631  | 0.393505  | 0.142857  | 00:53  |
| 1     | 0.326486  | 0.412136  | 0.157143  | 00:52  |
| 2     | 0.232185  | 0.388497  | 0.142857  | 00:51  |
| 3     | 0.207576  | 0.382458  | 0.142857  | 00:51  |


Great - The data cleaning has in this case lead to a worse model! But please note that there are far more way to clean or add to data to improve the error rate, one such method would be to perform data augmentation which will give the model considerably more images to work with - and as these have been augmented, the model should get better with worse images. 

The above said, I tested this newest model on a range of Tom Cruise photos and it didn't get a thing wrong! So with that, it's now time to export the model, ready for download into a Python script. This can be done as follows:

```python
learn.export('tcmodel.pkl')
```

## Deploying the Model

Upon exporting the model, we can download it into our local files. We'll sort the backend of our application out, whilst the front end can be sorted by a platform called [Gradio](https://www.gradio.app) using simple code that we can run through shortly. The hosting can then be covered by Hugging Face Spaces. To set this up you need to clone a Hugging Face Spaces repo into your VS Code Module and then we can get cracking!

First, in an 'app.py' file in VSCode, lets get down the following code:

```python
from fastai.vision.all import *
import gradio as gr

learner = load_learner('tcmodel.pkl')

categories = ('Deepfake Tom Cruise', 'Real Tom Cruise')

def classify_image(img):
    pred, idx, probs = learner.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['goodtcdeepfake.jpeg', 'badtcdeepfake.jpeg', 'realtc.jpeg']
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
```
This will set up our front end (don't be too hastey in committing this to the repo as there is some set up there to consider). Note that I have three examples, you can add three images to your file and route them accordingly. 

Next, you'll need to ensure that you have you 'tcmodel.pkl' in the file as well as a 'requirements.txt' file that says fastai on one line and gradio on the other. Your file should now have, the pkl model, the app.py, your three images and the requirements.txt file. Now it's time to commit and push to Hugging Face - but don't be too hastey! The model file is too big, and the images won't get pushed to Hugging Face, so you need to use LFS. 

In the command line, enter each of the following:

```cmd
git lfs install
git lfs track "*.jpg"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Configure Git LFS tracking"
```

This will make LFS track the pkl file and your image files (assuming they are jpg). Add and commit these with the following lines:

```cmd
git add cat.jpg dog.jpg model.pkl
git commit -m "Re-add binary files with Git LFS"
```

Once, they are committed, you should be in the clear so go ahead and and & commit the rest of the files and then push these to Hugging Face. The outcome is that you should have a app hosted on Hugging Face! [Here's mine](https://huggingface.co/spaces/mikeymoomin/tomcruise)!
