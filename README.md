# Super Image resolution

In this project we aim to use the dataset of face images and try to improve the resolution of blurred images using two approaches: <br>
<ul>
<li>Super Image Resolution Generative Adversarial Networks (SRGANs)</li>
<li>Autoencoders</li>

<b>Note</b>: The model trained using SRGANs is quite computationally heavy to train. Even the AWS EC2 instance for Nvidia Tesla T4 16GB GPU fell short in memory to train the model. There can be various approach like reducing the batch size and the quality of the network(generator network in GAN). But since the SRGAN approach is basically the research paper implementation I did not change any of these in order to make it easy for my model to train. The original research paper can be found <a href="https://arxiv.org/pdf/1609.04802.pdf">here</a>. 
<br>
The autoencoder approach was another project done by me on a different dataset, so yes, the dataset for each approach is different (and I'm sorry for that inconsistency :P). The autoenoder model training has 500 epochs and each epoch took 21 seconds to train on Nvidia Tesla T80 16GB GPU. 
<br>
I even desire to train a super resolution model using diffusion models. But I'm yet to understand its mathematics, so hold on tight, the diffusion model is up next to conquor! 
