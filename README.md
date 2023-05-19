# Energy Approximation
Out of curiosity, this repository explores the possibility of using deep learning approaches to approximate the minimum energy after RNA 2nd structure formation


# Traditional approaches
Traditional approaches for obtaining energy values rely on real-world experiments and Turner parameters, and these methods require significant time, resources, and experimental setups. Deep learning models have shown great promise in various domains, and utilizing them for obtaining energy values could potentially offer faster and more cost-effective alternatives

# Deep learning approaches
The deep learning approach tries to leverage the power of neural networks to approximate the minimum energy values of RNA 2nd structures. In this experiment I tried using a DistilBert model and trained it on a dataset that use RNA 1st sequences and 2nd structures as input, and their corresponding minimum energy values obtained through traditional methods (computed by Vienna RNA package) as predicted variable. Hopefully, we can teach the model to learn the underlying patterns and relationships.
## input:
Concatenate one-hotted RNA 1st sequence and one-hotted 2nd structure as input. Since they are one-hotted, I do not need to tokenize anymore, only embedding is needed when load into DistilBerts.
## output: 
Experimental energy values
## loss function: 
For simplicity, use MSE Loss

# Datasets
Datasets are obtained from open-source dataset: https://bprna.cgrb.oregonstate.edu/download.php, and I choose to use dot-bracket files.

After processing, the processed datasets turns out to be too large, follow the datapreprocessing.py steps to come up with the dataset.

# Predicted VS. Expreiment
![2711683609174_ pic](https://github.com/liangkunn/EnergyApproximation/assets/36016499/8d87e9f0-7fd3-4034-832b-42f0daa34c24)

The model does learn something useful as we can see there is a linear relationship between predicted and experimental energy vlaues. 

# Coefficient
![2741683609319_ pic](https://github.com/liangkunn/EnergyApproximation/assets/36016499/90239713-cfe4-4375-bde2-89d9214917c2)

R-square value is 0.81
