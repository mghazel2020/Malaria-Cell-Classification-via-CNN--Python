# Malaria Cells Classification using Convolutional Neural Networks (CNN)

<img align="center" src="images/banner-02.jpg" width="500" >

## 1. Objective

The objective of this project is to develop, train and evaluate the performance of a Convolutional Neural Networks (CNN) model to classify malaria cells into uninfected or infected classes.  


## 2. Motivation

Malaria remains a major burden on global health, especially in tropical countries, as illustrated in the figure below. Hundreds of millions of blood films are examined every year for malaria, which involves manual counting of parasites and infected red blood cells by a trained microscopist. Accurate parasite counts are essential not only for malaria diagnosis. They are also important for testing for drug-resistance, measuring drug-effectiveness, and classifying disease severity. However, microscopic diagnostics is not standardized and depends heavily on the experience and skill of the microscopist. It is common for microscopists in low-resource settings to work in isolation, with no rigorous system in place that can ensure the maintenance of their skills and thus diagnostic quality. This leads to incorrect diagnostic decisions in the field. For false-negative cases, this leads to unnecessary use of antibiotics, a second consultation, lost days of work, and in some cases progression into severe malaria. For false-positive cases, a misdiagnosis entails unnecessary use of anti-malaria drugs and suffering from their potential side effects, such as nausea, abdominal pain, diarrhea, and sometimes severe complications [1]. 

Recently, the application of machine and deep learning, for the classification of malarial cells has been explored and demonstrated to yield high accuracy [1-8]. In this work, we shall demonstrate such an application. In particular, we design, train and evaluate the performance of the convolutional network (CNN) to classify malarial cells.

<img align="center" src="images/malaria-disease.jpg" width="10000" >


## 3. Data

We make use of the open source Malaria Datasets made available by the National Institute of Medicine.  The data set consists of:

* 13,779 image patches of parasitized/infected cell
* 13,779 image patches of uninfected/healthy cells.

The  classification of the malaria cells using this data set involves the following 3 step: 

* Split this dataset into:
* Training data subset (80%)
* Testing data subsets (20%)
* Train the design CNN model on the training data subset
* Evaluate the performance of the trained model, in terms of classification accuracy, on the test data subset.

## 4. Development

In this section, we shall develop, train, deploy and evaluate the performance of a CNN model to classify malaria cells into parasitized (0) or uninfected (1).

* Author: Mohsen Ghazel (mghazel)
* Date: April 1st, 2021
* Project: Classification of Malarial Cells using Convolutional Neural Networks (CNN):
The objective of this project, we develop a Convolutional Network Model (CNN) to classify Malarial cells into:

  * Parasitized/infected cell (0)
  * Uninfected/healthy cell (1)
  
We shall demonstrate the end-to-end process, step by step.

### 4.1: Step 1: Python imports and global variables

#### 4.1.1. Imports:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">print<span style="color:#808030; ">(</span><span style="color:#074726; ">__doc__</span><span style="color:#808030; ">)</span>

Automatically created module <span style="color:#800000; font-weight:bold; ">for</span> IPython interactive environment
</pre>

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplot lib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># PIL library</span>
<span style="color:#800000; font-weight:bold; ">from</span> PIL <span style="color:#800000; font-weight:bold; ">import</span> Image
<span style="color:#696969; "># keras</span>
<span style="color:#800000; font-weight:bold; ">import</span> keras

<span style="color:#696969; "># sklearn imports</span>
<span style="color:#696969; "># - nededed for splitting the dataset into training and testing subsets</span>
<span style="color:#800000; font-weight:bold; ">from</span> sklearn<span style="color:#808030; ">.</span>model_selection <span style="color:#800000; font-weight:bold; ">import</span> train_test_split
<span style="color:#696969; "># - nededed for 1-hot coding of the image labels</span>
<span style="color:#800000; font-weight:bold; ">from</span> keras<span style="color:#808030; ">.</span>utils <span style="color:#800000; font-weight:bold; ">import</span> to_categorical

<span style="color:#696969; "># set the keras backend to tensorflow</span>
<span style="color:#696969; "># os.environ['KERAS_BACKEND'] = 'tensorflow'</span>
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># sys</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys

<span style="color:#696969; "># datetime</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#ffffff;font-size:10px;">Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Mount my Google drive to access the dataset</span>
<span style="color:#800000; font-weight:bold; ">from</span> google<span style="color:#808030; ">.</span>colab <span style="color:#800000; font-weight:bold; ">import</span> drive
drive<span style="color:#808030; ">.</span>mount<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'/content/drive'</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#ffffff;font-size:10px;">Drive already mounted at <span style="color:#44aadd; ">/</span>content<span style="color:#44aadd; ">/</span>drive<span style="color:#808030; ">;</span> 
to attempt to forcibly remount<span style="color:#808030; ">,</span> 
call drive<span style="color:#808030; ">.</span>mount<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"/content/drive"</span><span style="color:#808030; ">,</span> force_remount<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>
</pre>

#### 4.1.2. Global variables

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969;">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># We set the Numpy pseudo-random generator at a fixed value:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - This ensures repeatable results everytime you run the code. </span>
np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>seed<span style="color:#808030; ">(</span><span style="color:#008c00; ">101</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the random state to 101</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - This ensures repeatable results everytime you run the code. </span>
RANDOM_STATE <span style="color:#808030; ">=</span> <span style="color:#008c00; ">101</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Set the data directory where malaria data set is stored</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
image_directory <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'/content/drive/MyDrive/.../cell_images/'</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># set the input images size</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
INPUT_IMAGE_SIZE <span style="color:#808030; ">=</span> <span style="color:#008c00; ">64</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># We use the Malaria dataset from the National Librray of Medicine:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Source: </span><span style="color:#5555dd; ">https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html</span>
<span style="color:#696969; "># - It has 13,779 images of parasitized images</span>
<span style="color:#696969; "># - It has 13,779 images of uninfected images</span>
<span style="color:#696969; "># - Due to our limited computational resources, we shall only use a subset of </span>
<span style="color:#696969; ">#   this dataset</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># the number of used images from the parasitized and uninfected available data </span>
<span style="color:#696969; "># sets</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
NUM_USED_DATASET_IMAGES <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5000</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># The available labelled dataset is spit into subsets:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Testing data subset</span>
<span style="color:#696969; "># - Training data subset</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># This parameter indicates the proportion of randomly selected test data </span>
<span style="color:#696969; "># subset from the the full data set</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
TEST_DATA_SUBSET_PROPORTION <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.20</span> <span style="color:#696969; "># 20% of data is used for testing</span>
</pre>


### 4.2. Step 2: Read and visualize the input data set

We use the Malaria dataset from the National Library of Medicine:
Source: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html
It has 13,779 images of parasitized images
It has 13,779 images of uninfected images

Due to our limited computational resources, we shall only use a subset of this dataset:
NUM_USED_DATASET_IMAGES: number of images of parasitized images
NUM_USED_DATASET_IMAGES: number of images of uninfected images

In this section, we shall:
Read, resize, store and visualize the uninfected cells used images
Read, resize, store and visualize the infected cells used images

#### 4.2.1. Create data structures to store the read input data

Create lists to store:
The read and formatted input images
Their classification labels.



<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969;">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># Create containers to store the read images and their labels</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - create a list to store the images </span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
dataset <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">]</span> 

<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - create a list to store the labels, we use:</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; ">#   - 0: for parasitized images</span>
<span style="color:#696969; ">#   - 1: for uninfected images.</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
labels <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#808030; ">]</span>  
</pre>

#### 4.2.2. Read the infected/parasitized images:

Read and format the infected/parasitized images:
Each PNG is read and formatted and added to the dataset list
We visualize 25 images (every kth image)

##### 4.2.1.1. Iterate through all PNG images in the Parasitized sub-folder


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969;">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - Resize each image to: INPUT_IMAGE_SIZExINPUT_IMAGE_SIZE:</span>
<span style="color:#696969; ">#   - INPUT_IMAGE_SIZE: is global parameter</span>
<span style="color:#696969; "># - Save the resized into the dataset numpy array</span>
<span style="color:#696969; "># - Set its label to 0: Parasitized</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># get the images in the Parasitized sub-folder</span>
parasitized_images <span style="color:#808030; ">=</span> os<span style="color:#808030; ">.</span>listdir<span style="color:#808030; ">(</span>image_directory <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'Parasitized/'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># initialize the number of parasitized images</span>
num_parasitized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#696969; "># itererate over the content of the sub-folder</span>
<span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> image_name <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">enumerate</span><span style="color:#808030; ">(</span>parasitized_images<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>   
    <span style="color:#696969; "># check if the file extension is: png to indicate this is an image file</span>
    <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span>image_name<span style="color:#808030; ">.</span>split<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'.'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'png'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># increment the number of used parasitized images</span>
        num_parasitized_images <span style="color:#808030; ">=</span> num_parasitized_images <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># check if we have selected the NUM_USED_DATASET_IMAGES images</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> num_parasitized_images <span style="color:#44aadd; ">&gt;</span> NUM_USED_DATASET_IMAGES<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
          <span style="color:#696969; "># break</span>
          <span style="color:#800000; font-weight:bold; ">break</span><span style="color:#808030; ">;</span>

        <span style="color:#696969; "># read the image</span>
        image <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>imread<span style="color:#808030; ">(</span>image_directory <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'Parasitized/'</span> <span style="color:#44aadd; ">+</span> image_name<span style="color:#808030; ">)</span>
        <span style="color:#696969; "># format the image to type Image</span>
        image <span style="color:#808030; ">=</span> Image<span style="color:#808030; ">.</span>fromarray<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> <span style="color:#0000e6; ">'RGB'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># resiz ethe image</span>
        image <span style="color:#808030; ">=</span> image<span style="color:#808030; ">.</span>resize<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>INPUT_IMAGE_SIZE<span style="color:#808030; ">,</span> INPUT_IMAGE_SIZE<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># convert it back to numpy array and append to to dataset list</span>
        dataset<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span>image<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># append the label 0: Parasitized to labels list</span>
        labels<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># the final number of used parasitized images </span>
num_parasitized_images <span style="color:#808030; ">=</span> num_parasitized_images <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Read and formatted {0} parasitized input images</span><span style="color:#0f69ff; ">\n</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>num_parasitized_images<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Read <span style="color:#800000; font-weight:bold; ">and</span> formatted <span style="color:#008c00; ">5000</span> parasitized <span style="color:#400000; ">input</span> images
</pre>

##### 4.2.1.2. Visualize some of the Parasitized input images

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969;">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 25 parasitized images</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># set the number of skipped images</span>
<span style="color:#696969; "># - integer division</span>
NUM_SKIPPED_IMAGES <span style="color:#808030; ">=</span> num_parasitized_images <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">25</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">15</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">15</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Parasitized images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 25 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">25</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># create the subplot for the next ime</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>   
    <span style="color:#696969; "># image counter </span>
    image_counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span><span style="color:#400000; ">min</span><span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>i <span style="color:#44aadd; ">*</span> NUM_SKIPPED_IMAGES<span style="color:#808030; ">,</span> NUM_USED_DATASET_IMAGES <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>dataset<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image # "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>


#### 4.2.2. Read the uninfected/healthy images:

Read and format the uninfected/healthy images:
Each PNG is read and formatted and added to the dataset list
We visualize 25 images (every kth image)

##### 4.2.2.1. Iterate through all PNG images in the Uninfected sub-folder


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969;">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - Resize each image to: INPUT_IMAGE_SIZE x INPUT_IMAGE_SIZE:</span>
<span style="color:#696969; ">#   - INPUT_IMAGE_SIZE: is global parameter</span>
<span style="color:#696969; "># - Save the resized into the dataset numpy array</span>
<span style="color:#696969; "># - Set its label to 1: Uninfected</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># get the images in the Uninfected sub-folder</span>
uninfected_images <span style="color:#808030; ">=</span> os<span style="color:#808030; ">.</span>listdir<span style="color:#808030; ">(</span>image_directory <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'Uninfected/'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># initialize the number of uninfected images</span>
num_uninfected_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#696969; "># itererate over the content of the sub-folder</span>
<span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> image_name <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">enumerate</span><span style="color:#808030; ">(</span>uninfected_images<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>   
    <span style="color:#696969; "># check if the file extension is: png to indicate this is an image file</span>
    <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span>image_name<span style="color:#808030; ">.</span>split<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'.'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'png'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># increment the number of used uninfected images</span>
        num_uninfected_images <span style="color:#808030; ">=</span> num_uninfected_images <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># check if we have selected the NUM_USED_DATASET_IMAGES images</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span>num_uninfected_images <span style="color:#44aadd; ">&gt;</span> NUM_USED_DATASET_IMAGES<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
          <span style="color:#696969; "># break</span>
          <span style="color:#800000; font-weight:bold; ">break</span><span style="color:#808030; ">;</span>
          
        <span style="color:#696969; "># read the image</span>
        image <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>imread<span style="color:#808030; ">(</span>image_directory <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">'Uninfected/'</span> <span style="color:#44aadd; ">+</span> image_name<span style="color:#808030; ">)</span>
        <span style="color:#696969; "># format the image to type Image</span>
        image <span style="color:#808030; ">=</span> Image<span style="color:#808030; ">.</span>fromarray<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> <span style="color:#0000e6; ">'RGB'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># resiz e the image</span>
        image <span style="color:#808030; ">=</span> image<span style="color:#808030; ">.</span>resize<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>INPUT_IMAGE_SIZE<span style="color:#808030; ">,</span> INPUT_IMAGE_SIZE<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># convert it back to numpy array and append to to dataset list</span>
        dataset<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span>image<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># append the label 1: Uninfected to labels list</span>
        labels<span style="color:#808030; ">.</span>append<span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># the final number of used parasitized images </span>
num_uninfected_images <span style="color:#808030; ">=</span> num_uninfected_images <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Read and formatted {0} uninfected input images</span><span style="color:#0f69ff; ">\n</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>num_uninfected_images<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Read <span style="color:#800000; font-weight:bold; ">and</span> formatted <span style="color:#008c00; ">5000</span> uninfected <span style="color:#400000; ">input</span> images
</pre>

<img align="center" src="images/parasitized-25-images.jpg" width="10000" >

##### 4.2.2.2. Visualize some of the Uninfected input images

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># - Visualize 25 uninfected images</span>
<span style="color:#696969; ">#-----------------------------------------------------------------</span>
<span style="color:#696969; "># set the number of skipped images</span>
<span style="color:#696969; "># - integer division</span>
NUM_SKIPPED_IMAGES <span style="color:#808030; ">=</span> num_uninfected_images <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">25</span>
<span style="color:#696969; "># specify the overall grid size</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">15</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">15</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span> 
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Uninfected images"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the 25 images</span>
<span style="color:#800000; font-weight:bold; ">for</span> i <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">25</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># create the subplot for the next ime</span>
    plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>   
    <span style="color:#696969; "># image counter </span>
    image_counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span><span style="color:#400000; ">min</span><span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>NUM_USED_DATASET_IMAGES <span style="color:#44aadd; ">+</span> 
                                     NUM_SKIPPED_IMAGES <span style="color:#44aadd; ">+</span> 
                                     i <span style="color:#44aadd; ">*</span> NUM_SKIPPED_IMAGES<span style="color:#808030; ">,</span> 
                                     <span style="color:#008c00; ">2</span> <span style="color:#44aadd; ">*</span> NUM_USED_DATASET_IMAGES <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span> <span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>dataset<span style="color:#808030; ">[</span>image_counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image # "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>image_counter<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
</pre>

<img align="center" src="images/uninfected-25-images.jpg" width="10000" >


### 4.3. Step 3: Split the dataset into training and testing data subsets

Split the dataset into training and testing dataset:
* Testing data subset proportion: TEST_DATA_SUBSET_TEST_DATA_SUBSET_PROPORTION
* Training data subset fraction: (1-TEST_DATA_SUBSET_TEST_DATA_SUBSET_PROPORTION)

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#696969; "># Split the dataset into:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Testing data subset: TEST_DATA_SUBSET_PROPORTION</span>
<span style="color:#696969; "># - Training data subset: (1-TEST_DATA_SUBSET_PROPORTION).</span>
X_train<span style="color:#808030; ">,</span> X_test<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">,</span> y_test <span style="color:#808030; ">=</span> train_test_split<span style="color:#808030; ">(</span>dataset<span style="color:#808030; ">,</span> to_categorical<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span>labels<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                                                    test_size <span style="color:#808030; ">=</span> TEST_DATA_SUBSET_PROPORTION<span style="color:#808030; ">,</span> 
                                                    random_state <span style="color:#808030; ">=</span> RANDOM_STATE<span style="color:#808030; ">)</span>
<span style="color:#696969; "># display message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The dataset is split into training and testing subsets:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The number of training images = "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The number of test images = "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The dataset <span style="color:#800000; font-weight:bold; ">is</span> split into training <span style="color:#800000; font-weight:bold; ">and</span> testing subsets<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The number of training images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8000</span>
The number of tst images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">2000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

### 4.4. Step 4: Build the CNN model

Build the CNN model:
A sequence of convolutional and pooling layers.
With some some normalization and dropout layers in between.
Experiment with different structures and hyper parameters

#### 4.4.1. Define the model layers:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Define sequential layers of the CNN model:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 1) Input layer with image size: INPUT_IMAGE_SIZE x INPUT_IMAGE_SIZE</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># the input layer size</span>
INPUT_SHAPE <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span>INPUT_IMAGE_SIZE<span style="color:#808030; ">,</span> INPUT_IMAGE_SIZE<span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>   
<span style="color:#696969; "># create the input layer</span>
inp <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span><span style="color:#400000; ">Input</span><span style="color:#808030; ">(</span>shape<span style="color:#808030; ">=</span>INPUT_SHAPE<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 2) Convolutional layer # 1: with 32 filters</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
conv1 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> kernel_size<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                            activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>inp<span style="color:#808030; ">)</span>
<span style="color:#696969; "># max-pooling layer</span>
pool1 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>MaxPooling2D<span style="color:#808030; ">(</span>pool_size<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>conv1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># batch-normalization layer</span>
norm1 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>BatchNormalization<span style="color:#808030; ">(</span>axis <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>pool1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># dropout layer</span>
drop1 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span>rate<span style="color:#808030; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>norm1<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 3) Convolutional layer # 2: with 32 filters</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------</span>
conv2 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> kernel_size<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                               activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">,</span> padding<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'same'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>drop1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># max-pooling layer</span>
pool2 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>MaxPooling2D<span style="color:#808030; ">(</span>pool_size<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>conv2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># batch-normalization layer</span>
norm2 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>BatchNormalization<span style="color:#808030; ">(</span>axis <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>pool2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># dropout layer</span>
drop2 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span>rate<span style="color:#808030; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>norm2<span style="color:#808030; ">)</span>

<span style="color:#696969; "># Flatten the matrix to get it ready for dense layers</span>
flat <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Flatten<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>drop2<span style="color:#808030; ">)</span>  

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 4) Dense layer # 1: with 512 neurons</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># dense layer of size 512 neurons</span>
hidden1 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">512</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>flat<span style="color:#808030; ">)</span>
<span style="color:#696969; "># batch-normalization layer</span>
norm3 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>BatchNormalization<span style="color:#808030; ">(</span>axis <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>hidden1<span style="color:#808030; ">)</span>
<span style="color:#696969; "># dropout layer</span>
drop3 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span>rate<span style="color:#808030; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>norm3<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 5) Dense layer # 2: with 256 neurons</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
hidden2 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">256</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>drop3<span style="color:#808030; ">)</span>
<span style="color:#696969; "># batch-normalization layer</span>
norm4 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>BatchNormalization<span style="color:#808030; ">(</span>axis <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>hidden2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># dropout layer</span>
drop4 <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span>rate<span style="color:#808030; ">=</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>norm4<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># 6) Output layer: with 2 outputs</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># final output layer with 2 outputs (0 vs. 1)</span>
out <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'sigmoid'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>drop4<span style="color:#808030; ">)</span>  
</pre>

#### 4.4.2. Construct the Keras model using the above defined layers:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Define the Keras model using the above defined layers:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; ">#  - Experiment with using:</span>
<span style="color:#696969; ">#      - binary_crossentropy: suitable for binary classification </span>
<span style="color:#696969; ">#      - categorical_crossentropy: suitable for multi-class classification </span>
model <span style="color:#808030; ">=</span> keras<span style="color:#808030; ">.</span>Model<span style="color:#808030; ">(</span>inputs<span style="color:#808030; ">=</span>inp<span style="color:#808030; ">,</span> outputs<span style="color:#808030; ">=</span>out<span style="color:#808030; ">)</span>
</pre>

4.4.3. Compile the CNN model:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compile the model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model<span style="color:#808030; ">.</span><span style="color:#400000; ">compile</span><span style="color:#808030; ">(</span>optimizer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'adam'</span><span style="color:#808030; ">,</span>
                loss<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'categorical_crossentropy'</span><span style="color:#808030; ">,</span>   
                metrics<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
</pre>

#### 4.4.4. Print the model summary



<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Printout the model summary</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># print model summary</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>model<span style="color:#808030; ">.</span>summary<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Model<span style="color:#808030; ">:</span> <span style="color:#0000e6; ">"model_3"</span>
_________________________________________________________________
Layer <span style="color:#808030; ">(</span><span style="color:#400000; ">type</span><span style="color:#808030; ">)</span>                 Output Shape              Param <span style="color:#696969; ">#   </span>
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
input_4 <span style="color:#808030; ">(</span>InputLayer<span style="color:#808030; ">)</span>         <span style="color:#808030; ">[</span><span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">]</span>       <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d_6 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">896</span>       
_________________________________________________________________
max_pooling2d_6 <span style="color:#808030; ">(</span>MaxPooling2 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">0</span>         
_________________________________________________________________
batch_normalization_12 <span style="color:#808030; ">(</span>Batc <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">128</span>       
_________________________________________________________________
dropout_12 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>         <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d_7 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">9248</span>      
_________________________________________________________________
max_pooling2d_7 <span style="color:#808030; ">(</span>MaxPooling2 <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">0</span>         
_________________________________________________________________
batch_normalization_13 <span style="color:#808030; ">(</span>Batc <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">128</span>       
_________________________________________________________________
dropout_13 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>         <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">16</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">0</span>         
_________________________________________________________________
flatten_3 <span style="color:#808030; ">(</span>Flatten<span style="color:#808030; ">)</span>          <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8192</span><span style="color:#808030; ">)</span>              <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_9 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">4194816</span>   
_________________________________________________________________
batch_normalization_14 <span style="color:#808030; ">(</span>Batc <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">2048</span>      
_________________________________________________________________
dropout_14 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>         <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_10 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">131328</span>    
_________________________________________________________________
batch_normalization_15 <span style="color:#808030; ">(</span>Batc <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">1024</span>      
_________________________________________________________________
dropout_15 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>         <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">256</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_11 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>             <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>                 <span style="color:#008c00; ">514</span>       
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
Total params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">340</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">130</span>
Trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">338</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">466</span>
Non<span style="color:#44aadd; ">-</span>trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">664</span>
_________________________________________________________________
<span style="color:#074726; ">None</span>
</pre>

### 4.5. Step 5: Fit/train the model

* Train the model on the training data set


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># train the model</span>
history <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span>X_train<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>            <span style="color:#696969; "># training data images</span>
                         y_train<span style="color:#808030; ">,</span>                 <span style="color:#696969; "># training data labels</span>
                         batch_size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span>         <span style="color:#696969; "># experiment with the batch-size</span>
                         verbose <span style="color:#808030; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span>             <span style="color:#696969; "># logging-flag: set 0, 1, 2, 3, etc.</span>
                         epochs <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#808030; ">,</span>            <span style="color:#696969; "># experiment with the number of epochs</span>
                         validation_split <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.10</span><span style="color:#808030; ">,</span>  <span style="color:#696969; "># fraction of the validation data subset</span>
                         shuffle <span style="color:#808030; ">=</span> <span style="color:#074726; ">False</span>          <span style="color:#696969; "># set to False</span>
                         <span style="color:#696969; "># callbacks=callbacks    # we did not implement any callbacks function</span>
                     <span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">37</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6653</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7049</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">4.0594</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6025</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4408</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8042</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7894</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7513</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3525</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8475</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.0265</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6575</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.2286</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9112</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">2.9734</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5562</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1610</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9388</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">2.0291</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6150</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
Epoch <span style="color:#008c00; ">95</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0053</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9981</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5435</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9388</span>
Epoch <span style="color:#008c00; ">96</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0055</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9976</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.7706</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9212</span>
Epoch <span style="color:#008c00; ">97</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0054</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9982</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5282</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9438</span>
Epoch <span style="color:#008c00; ">98</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0028</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9990</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3392</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9550</span>
Epoch <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0039</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9992</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3415</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9500</span>
Epoch <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">113</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">113</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">35</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0012</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9997</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3898</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9550</span>
</pre>

### 4.6. Step 6: Evaluate the model

Evaluate the performance of the trained model on the test data subset

#### 4.6.1. Model training performance metrics: Loss function

Display the variations of:

* The loss function
* The classification accuracy for the different epochs for:
   * The training data subset
   * The validation data subset


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">f<span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>ax1<span style="color:#808030; ">,</span> ax2<span style="color:#808030; ">)</span> <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
t <span style="color:#808030; ">=</span> f<span style="color:#808030; ">.</span>suptitle<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'CNN Performance: Malarial Cell Classification - 0: Parasitized, 1: Uninfected'</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">14</span><span style="color:#808030; ">)</span>
f<span style="color:#808030; ">.</span>subplots_adjust<span style="color:#808030; ">(</span>top<span style="color:#808030; ">=</span><span style="color:#008000; ">0.85</span><span style="color:#808030; ">,</span> wspace<span style="color:#808030; ">=</span><span style="color:#008000; ">0.3</span><span style="color:#808030; ">)</span>

max_epoch <span style="color:#808030; ">=</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>history<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span>
epoch_list <span style="color:#808030; ">=</span> <span style="color:#400000; ">list</span><span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span>max_epoch<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
ax1<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>epoch_list<span style="color:#808030; ">,</span> history<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Train Accuracy'</span><span style="color:#808030; ">)</span>
ax1<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>epoch_list<span style="color:#808030; ">,</span> history<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Validation Accuracy'</span><span style="color:#808030; ">)</span>
ax1<span style="color:#808030; ">.</span>set_xticks<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> max_epoch<span style="color:#808030; ">,</span> <span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
ax1<span style="color:#808030; ">.</span>set_ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy Value'</span><span style="color:#808030; ">)</span>
ax1<span style="color:#808030; ">.</span>set_xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch'</span><span style="color:#808030; ">)</span>
ax1<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy'</span><span style="color:#808030; ">)</span>
l1 <span style="color:#808030; ">=</span> ax1<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span>loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"best"</span><span style="color:#808030; ">)</span>

ax2<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>epoch_list<span style="color:#808030; ">,</span> history<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Train Loss'</span><span style="color:#808030; ">)</span>
ax2<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>epoch_list<span style="color:#808030; ">,</span> history<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Validation Loss'</span><span style="color:#808030; ">)</span>
ax2<span style="color:#808030; ">.</span>set_xticks<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> max_epoch<span style="color:#808030; ">,</span> <span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
ax2<span style="color:#808030; ">.</span>set_ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss Value'</span><span style="color:#808030; ">)</span>
ax2<span style="color:#808030; ">.</span>set_xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch'</span><span style="color:#808030; ">)</span>
ax2<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss'</span><span style="color:#808030; ">)</span>
l2 <span style="color:#808030; ">=</span> ax2<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span>loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"best"</span><span style="color:#808030; ">)</span>
</pre>

<img align="center" src="images/performance-metrics.jpg" width="10000" >

#### 4.6.2. Overall Accuracy on test data subset:

Compute the overall accuracy on the test data subset:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Accuracy calculation on the test data subset</span>
test_data_accuracy <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>evaluate<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span>X_test<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>array<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#44aadd; ">*</span><span style="color:#008c00; ">100</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Trained model performance evaluation on test data subset:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test_Accuracy: {:.2f}%"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>test_data_accuracy<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"-------------------------------------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#008c00; ">63</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">63</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">42</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3806</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9565</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Trained model performance evaluation on test data subset<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test_Accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">95.65</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

### 4.7. Step 7: Save the trained CNN model:

* Save the trained model for future re-ue:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Save the trained model</span>
model<span style="color:#808030; ">.</span>save<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'trained-malaria-cells-classification-cnn-model-01Apr2021.h5'</span><span style="color:#808030; ">)</span>
</pre>

### 4.8. Step 8: End of Execution:

* Display a successful end of execution message:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">02</span> <span style="color:#008c00; ">01</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">43</span><span style="color:#808030; ">:</span><span style="color:#008000; ">43.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>

## 5. Analysis

In view of the presented results, we make the following observations:

* The trained CNN model yields significantly accurate classification of the malarial cells, even though we have only used a subset of 10,000 images of the available 2x 13,779 = 27,598 images.
* Higher accuracy may be achieved using all of the available images.

## 6. Future Work

We propose to explore the following related issues:

* To explore hyper-parameters fine-tuning of the designed model to improve performance
* To explore changes to the designed model by adding new layers and configurations parameters
* To use the full available data set of 27,598 images to retrain and evaluate the model performance.

## 7. References

1. National Library of Medicine. Malaria Datasets. Retrieved from: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html
2. Poostchi, M. et al. (April 1st, 2021). Image analysis and machine learning for detecting malaria. Retrieved from:  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5840030/ (September 2th, 2022).
3. Towards data science. (April 1st, 2021).  Deep learning to identify Malaria cells using CNN on Kaggle. Retrieved from:  https://towardsdatascience.com/deep-learning-to-identify-malaria-cells-using-cnn-on-kaggle-b9a987f55ea5 (September 2th, 2022).
4. Pan, D. et al. (April 1st, 2021). Classification of Malaria-Infected Cells Using Deep Convolutional Neural Networks. Retrieved from:  https://www.intechopen.com/books/machine-learning-advanced-techniques-and-emerging-applications/classification-of-malaria-infected-cells-using-deep-convolutional-neural-networks (September 2th, 2022).
5. Kaggle. (April 1st, 2021). Malaria Cell Images Classification - CNN. Retrieved from:  https://www.kaggle.com/mrudhuhas/malaria-cell-images-classification-cnn
6. MARKTECHPOST. (April 1st, 2021). Recognizing Malaria Cells Using Keras Convolutional Neural Network(CNN). Retrieved from:  https://www.marktechpost.com/2019/12/09/recognizing-malaria-cells-using-keras-convolutional-neural-networkcnn/ (September 2th, 2022).
87. Rahman, A. et all. (April 1st, 2021).  Improving Malaria Parasite Detection from Red Blood Cell using Deep Convolutional Neural Networks. Retrieved from:  https://arxiv.org/ftp/arxiv/papers/1907/1907.10418.pdf (September 2th, 2022).
8. Pattanaik, P. (April 1st, 2021). Deep CNN Frameworks Comparison for Malaria Diagnosis. Retrieved from:  https://arxiv.org/ftp/arxiv/papers/1909/1909.02829.pdf (September 2th, 2022).








