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

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Dprint<span style="color:#808030; ">(</span><span style="color:#074726; ">__doc__</span><span style="color:#808030; ">)</span>

Automatically created module <span style="color:#800000; font-weight:bold; ">for</span> IPython interactive environment
</pre>
