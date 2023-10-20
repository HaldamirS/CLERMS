## CLERMA
CLERMS is a novel contrastive learning-based method for the representation of MS/MS spectra, which is based on transformer architecture. The model architecture equipped with a sinusoidal embedder and a novel loss function composed of InfoNCE loss and MSE loss has been proposed for the obtaining of good embedding from the peak information and the metadata. comparison between our method and our methods demonstrates that our model can get a better performance in several specific tasks.


## requirements
-  pandas              1.4.3  
-  torch               1.12.0+cu113  
-  numpy               1.19.4  
-  scikit-learn        1.2.0  
-  scipy               1.4.1  
-  seaborn             0.12.2  
-  matchms             0.18.0
-  matplotlib          3.5.2
-  swifter             1.3.4
-  tqdm                4.51.0
-  tensorflow          2.3.0
-  rdkit               2022.9.3

## data preparation
In this paper, we use GNPS to verify the performance of our model on compound identification and spectra clustering. The GNPS data set can be obtained from https://zenodo.org/record/5186176, and should be put to the directory `data`. 

## data preprocessing
Some of the records in the spectra data contain inaccurate data or some of the information is missing. So, we remove them from the input data. Also, the peak information needs to be normalized for the model input. In this process， just run `dataset_preprocessing.ipynb`. 

To get the structural similarity for the model training, we calculate the score from the input data. In this process, just run `cal_tanimoto_score.ipynb`.


## model training 
model training and embedding obtaining are introduced in this step. All the parameters for the training can be modified in this step. To get the embedding, just run `model.ipynb`.

## note
if you find this code helpful and use it, please cite our paper. Any question about our paper, please contact with sunhaiming@hikvision.com