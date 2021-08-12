# cleanify

### Intro
This project comprises two tools: The `dirtify` Python tool takes a bunch of images and grabs small square thumbnails of them, then performs some sort of dirty filter on them, from a list of options. The `cleanify` tool uses a deep convolutional neural network to try to learn to convert dirty images to clean images.
### Dataset
The required folder and dataset structure are:
<pre>
├───source
│   cleanify.py
│   dirtify.py
└───input
    └───raw
</pre>

You'll need to populate the raw folder with a bunch of source images. In our initial experiments, we used the Sharp subset of the [Blur dataset](https://www.kaggle.com/kwentar/blur-dataset "Blur dataset").

##### dirtify
To experiment with the Cleanify project, set your working directory to the source folder and run the dirtify tool. Then run the cleanify tool. Here are some examples:

`dirtify --tile --blur 15` (separates source images into tiles, then does a 15-pixel Gaussian blur)</br>  
`dirtify --jpeg 50` (compresses with JPEG at 50% quality, then decompresses)</br>
`dirtify --blur 5` (performs a Gaussian blur with a 5-pixel radius)</br>
`dirtify --noise 25` (renders 25% uniform noise on image)</br>
`dirtify --invert` (inverts the pixels)</br>
`dirtify --xout` (draws a dark red 1-pixel-wide "X" through the image)</br>
`dirtify -j 80 -b 3 -i -x` (Does a little of everything)</br>

The `dirtify` process will create new sibling folders beside raw:

<pre>
└───input
    ├───raw
    ├───clean
    └───dirty
</pre>

##### cleanify
`cleanify` (by default, applies 40 epochs)</br>
`cleanify --epochs 100` (you can specify a number of epochs)</br>
`cleanify --autoencoder` (you can use an Autoencoder NN rather than a vanilla CNN)</br>

The results will appear in a new output folder.
<pre>
├───source
│   cleanify.py
│   dirtify.py
├───input
│   ├───raw
│   ├───clean
│   └───dirty
└───output
    └───saved_images
</pre>

### History
This project was derived from Sovit Ranjan Rath's tutorial, [Image Deblurring using Convolutional Neural Networks and Deep Learning](https://debuggercafe.com/image-deblurring-using-convolutional-neural-networks-and-deep-learning "Tutorial") and his original [GitHub repository](https://github.com/sovit-123/image-deblurring-using-deep-learning "Repo").
