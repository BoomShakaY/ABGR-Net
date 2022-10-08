This repository provides the code for the paper entitled [Auxiliary Bi-Level Graph Representation for Cross-Modal Image-Text Retrieval](https://ieeexplore.ieee.org/document/9428380), and accepted for publication in the 2021 IEEE International Conference on Multimedia and Expo (ICME 2021). 

The files contain all the core algorithms of ABGR-Net. 

### Acknowledgment:
Some part of our implementation is built on top of the opened source code of [SCAN](https://github.com/kuanghuei/SCAN), where you can find part of instructions.


### Requirments:
- python == 2.7
- torch == 1.2.0
- trochvision == 0.4.0
- TensorBoard
- h5py
- Punkt Sentence Tokenizer:
```
import nltk
nltk.download()
> d punkt
```

Detailed training and testing instructions will be coming soon......


### Cite:
In case our project helps your research, we appreciate it if you cite it in you works.

```
@INPROCEEDINGS{Zhong2021Auxiliary,
  author    = {Zhong, Xian and Yang, Zhengwei and Ye, Mang and Huang, Wenxin and Yuan, Jingling and Lin, Chia-Wen},
  booktitle = {2021 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title     = {Auxiliary Bi-Level Graph Representation for Cross-Modal Image-Text Retrieval}, 
  year      = {2021},
  pages     = {1-6},
  doi       = {10.1109/ICME51207.2021.9428380}}
```
