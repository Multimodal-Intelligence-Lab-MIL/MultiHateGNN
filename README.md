# [Multimodal Hate Detection Using Dual-Stream Graph Neural Networks](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_408/paper.pdf)
![](https://github.com/Multimodal-Intelligence-Lab-MIL/MultiHateGNN/blob/main/pipeline.png)

Hateful videos present serious risks to online safety and real-world well-being, necessitating effective detection methods. Although multimodal classification approaches integrating information from several modalities outperform unimodal ones, they typically neglect that even minimal hateful content defines a video’s category. Specifically, they generally treat all content uniformly, instead of emphasizing the hateful components. Additionally, existing multimodal methods cannot systematically capture essential structured information in videos, which limits the effectiveness of multimodal fusion. To address these limitations, we propose a novel classification model, the multimodal dual stream graph neural networks. It constructs an instance graph by separating the given video into several instances to extract instance-level features. Then, a complementary weight graph assigns importance weights to these features, highlighting hateful instances.
Importance weights and instance features are combined to generate video labels. Our model employs a graph-based framework to systematically model structured relationships within and across modalities. Extensive experiments on public datasets show that our model is state-of-the-art in hateful video classification and has strong explainability.

## Get Started
### Dependencies
Below is the key environment with the recommended version under which the code was developed:   
Python 3.8; torch 2.0.0; numpy 1.22.3; Cuda 11.1  

### Training and Evaluation
The training and evaluation can be implemented by running the scripts/train.py. GNN.py and models_tran_feature.py in the model folder provide the structures of the employed GNNs, LSTMs, and MLPs. dataset.py shows the data processing, while the GNNDual.py in the cfgs folder gives the experimental parameters.

### Authors  
Jiangbei Yue, Shuonan Yang, Tailin Chen, Jianbo Jiao, Zeyu Fu

### Acknowledgement 
The research work was supported by the Alan Turing Institute and DSO National Laboratories Framework Grant Funding.

### Citation (Bibtex)  
Please cite our paper if you find it useful:
```
@article{yue2025multimodal,
  title={Multimodal Hate Detection Using Dual-Stream Graph Neural Networks},
  author={Yue, Jiangbei and Yang, Shuonan and Chen, Tailin and Jiao, Jianbo and Fu, Zeyu},
  booktitle={The 36th British Machine Vision Conference},
  year={2025}
}
```
