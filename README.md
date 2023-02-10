<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />

<div align="center">
 <a href="https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI">
   <img src="images/logo.png" alt="Logo" width="80" height="80">
 </a>

 <h3 align="center">Exploring the Power of PCA and Machine Learning for Speech Emotion Recognition with a Focus on Explainable AI</h3>

 <p align="center">
   This paper has been submitted for publication in Canadian Conference on Artificial Intelligence (CANAI).
   <br />
  </p>
  <!-- <a href="https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI"><strong>Explore the docs »</strong></a> -->
</div>  

 
<div align="center">

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI/#readme "Go to project documentation")

</div>  

<div align="center">
   <p align="center">
   ·
   <a href="https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI/issues">Report Bug</a>
   ·
   <a href="https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI/issues">Request Feature</a>
 </p>
</div>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#getting-the-code">Getting the code</a></li>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#reproducing-the-results">Reproducing the results</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#on-tess-dataset">On TESS dataset</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABSTRACT -->
## Abstract

<p align="justify"> We present a new method for speech emotion recognition (SER) based on carefully engineered features of the voice and their statistical characteristics values. We pay particular attention to the explainability of our results, using methods for estimating the importance of individual features in a classification machine learning model in order to refine our features selection process. This, in turn, enhances the overall performance of our SER system. On the other hand, randomly computing features without considering their importance in SER can lead to including irrelevant or redundant information, which can reduce the performance of the model and lead to less accurate predictions. It also can make the model more complex and computationally expensive, leading to longer processing times and increased memory usage. Therefore, it is important to carefully consider and analyze the features used in SER systems in order to build models that are able to accurately and reliably predict emotions from speech signals. To do this, we first compute a set of speech features believed to be useful for emotion recognition. Then, we use principal component analysis (PCA) to select the best combinations of features that capture the most information in the data. The final feature set consists of the principal components of each selected combination. We train and fine tune an Extra Trees classifier on the refined feature set to obtain robust predictions. Additionally, we incorporate explainable artificial intelligence (XAI) methods to understand the role each feature plays in the predictions. By using Shapley values, we can optimize the feature selection process and improve the overall performance of the SER system through an iterative feedback loop. Our method outperforms state-of-the-art approaches as we provide a comprehensive way in the area of SER and XAI that aims to balance the benefits of advanced machine learning techniques with the need for transparency and comprehensibility for better results. The proposed method is evaluated on TESS dataset. </p>
<div align="center">
  
![model-architecture][model-architecture]
  
*Proposed method diagram*
  
</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[model-architecture]: images/XAI_1.png

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
* ![PyCaret](https://img.shields.io/badge/PyCaret-%23036CFF.svg?style=for-the-badge&logo=PyCaret&logoColor=white)
* ![Kaldi](https://img.shields.io/badge/Kaldi-%232465A0.svg?style=for-the-badge&logo=Kaldi&logoColor=white)
* ![SHAP](https://img.shields.io/badge/SHAP-%23006400.svg?style=for-the-badge&logo=SHAP&logoColor=white)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
<p align="justify">
We first unify sampling rate of the audio data to 16 KHz and using a mono-channel format. This ensures that the audio signals could be properly processed and analyzed by our system.
After going through the PCA-driven feature extraction and selection process, we use stratified random sampling to divide both the original dataset and the constructed dataset into three homogeneous groups, or "strata": training, validation, and testing. We hold back 10% of the data as unseen data to be used later for testing, 70% of the remaining data for training, and 30% for validation. This approach ensures that the distribution of classes is maintained across all datasets. Then, we use 10-fold cross-validation to train 14 machine learning models on each dataset to select the optimal one. By using cross-validation, we can obtain an estimate of the performance of the models that is less sensitive to the particular random partition of the data. Thus, we are able to better compare the contribution of the data selection technique of the most informative features and how much the performance improves.
In order to improve the performance of our best-performing machine learning model, we use the technique called grid search technique. This involves exhaustively searching through a specified parameter space to find the best combination of hyperparameters for a given model. In this way, we are able to fine-tune the model by adjusting its hyperparameters and making it more robust. The goal of this process is to find the optimal set of hyperparameters that produces the highest performance on the validation dataset.
After the grid search process is finished, we assess the performance of the final model by applying it to the 10% of the dataset that was set aside at the beginning of the experiment. This portion of the data serves as a test set to evaluate the model's ability to generalize to unseen data and estimate its generalization error. The testing performance is an indicator of how well the model would perform on new, unseen data. By using this method, we were able to ensure that our model is not overfitting to the training data. Finally, we use the SHAP approach for the explainable artificial intelligence module to evaluate the feature importance in the predictions made by our optimal model. This allows us to investigate how the model is making its predictions and identify which features are most important for determining the emotion. The metrics we use to evaluate our work are accuracy, recall, precision, and F1-Score.  
</p>

### Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI.git

or [download a zip archive](https://github.com/alaaNfissi/Exploring-the-Power-of-PCA-and-Machine-Learning-for-Speech-Emotion-Recognition-with-a-Focus-on-XAI/archive/refs/heads/main.zip).
