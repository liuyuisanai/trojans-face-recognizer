# 'Trojans' Face Recognizer

**Update 10/31/2019** We upload the ```loss.py``` as our implementation of the original ```ArcFace``` and the proposed ```ArcNegFace```. Now you can use them to train the model and compare the results. If you use the ms1m-v3 provided by the LFR challenge, you would reproduce the reported results as listed in our paper.

This is the model and code for the paper ['Towards Flops-constrained Face Recognition'](https://arxiv.org/pdf/1909.00632.pdf), which win the 1st place in the [ICCV19 Lightweight Face Recognition Challenge](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/), large video track.
This repo only covers the two 30Gflops tracks (deepglint-large and iQiyi-large) that we took participate in.
For more details such as network design, training strategy and bag of tricks, please refer to our [workshop paper](https://arxiv.org/pdf/1909.00632.pdf).

### Download code and models
 1. Using this [link](https://drive.google.com/open?id=1NhjPmPHkykrvxCCo3kz-ssPsv6ZaHhRp) to download the EfficientPolyFace and QAN++ models.
 2. Clone this repo and override the files with the same name.
### Efficient PolyFace for image face recognition (deepglint-large): 
 - Use this example to generate the representation (feature vector) for a single image: 
```
python test_tp.py --test_img 00000f9f87210c8eb9f5fb488b1171d7.jpg --save_path ./
```
### Enhanced quality aware network (QAN++) for video face recognition (iQiyi-large)
 1. Use this example to generate the representation and the quality for a single image. This will produce two files, which store the representation and quality respectively:
```
python test_quality.py --test_img 000005_99.jpg
```
Note that to generate quality for a set/sequence/video, you need to generate representations and qualities for all frames in this step by a loop.
 
 2. Use ```generate_aggregation_feat.py``` to generate presentations for a video.

### Reference
Please cite our paper if this project helps your research.
```
@inproceedings{liu2019towards,
  title={Towards Flops-constrained Face Recognition},
  author={Liu, Yu and Song, Guanglu and Zhang, Manyuan and Liu, Jihao and Zhou, Yucong and Yan, Junjie},
  booktitle={Proceedings of the ICCV Workshop},
  year={2019}
}
```
