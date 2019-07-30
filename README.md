# ShuffleNet v2 in Pytorch
- The paper about shufflenetv2: [shufflenet v2](https://arxiv.org/pdf/1807.11164.pdf)
- I implement the shufflenetv2, and test the performance on classification and detection tasks. You can use these codes to train model on your dataset.

## Version
- Python 3.6
- torch 1.1.0
- torchvision 0.3.0
## To do
- [x] Network structure
- [x] Used for Classification
- [x] Used for [Objection detection](https://github.com/ZhuYun97/ShufflNetv2-YOLOv3)
## Usage
### For classification
- If you want to test classification demo.
1. Download the project `https://github.com/ZhuYun97/shufflenetv2.git`
2. You should download the [dataset](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).
3. Then, put the dataset into the corresponding location.
#### Training
- Training from a fresh start: run the command `python main.py`
- Training from a trained model: `python main.py -t 1 -pre "/save/1563186508.pkl"`
#### Testing
- `python main.py -t 0 -pre "./save/1563186508.pkl"`
#### Explanation for some obscure arguments
- `t` means you decide to train(1) or test(0) the model. If you assign 0 to `t`, it's better to assign a trained model to `pre` 
- `e` means how many epochs
- `bs` means batch size

#### Comparision with other models
As for classification task
batch_size=1, GPU=1080

| Type | Acc | Time | MFLOPs |
| --- | --- | --- | --- |
| EfficientNet-B3 | 94.8 | 61.7 FPS | 1800 |
| ShuffleNet v2 | 94.7 | 145.6 FPS | 146 |
| MobileNet v2 | 94.3 | 190.4 FPS | 300 |
| MobileNet v3-Large | 89.8 | 109.8 FPS | 219 |
| MobileNet v3-Small | 90.9 | 126.2 FPS | 66 |

当在concat之后再se
On val dataset loss: 0.3544557321468989, acc: 0.9209333333333334
277.87629527741353 FPS

> mobilenetv2 1563588345.pkl
> shufflenetv2 1564211174.pkl

In additions, MobileNet v2 is hard to converge(training around 30 epochs).
> The dataset is dogvscat. And The GPU is 1080.

### For detection
In another [repo](https://github.com/ZhuYun97/ShufflNetv2-YOLOv3)
## Training own your dataset
### For classification
TBD
### For detection
In another [repo](https://github.com/ZhuYun97/ShufflNetv2-YOLOv3)
## Experiments
I train the model about 5 eopchs, and in each eopch, I test the performance of trained model.
```
Phase train loss: 0.6354673637662616, acc: 0.6564571428571429
Phase val loss: 0.5708242939949035, acc: 0.7146666666666667
Phase train loss: 0.493809922170639, acc: 0.7606285714285714
Phase val loss: 0.5668963393211365, acc: 0.724
Phase train loss: 0.4324655994551522, acc: 0.7994857142857142
Phase val loss: 0.4208303438186646, acc: 0.8048
Phase train loss: 0.38515312327657425, acc: 0.8273714285714285
Phase val loss: 0.37815397882064183, acc: 0.8298666666666666
Phase train loss: 0.3477836193084717, acc: 0.8467428571428571
Phase val loss: 0.34451772966384886, acc: 0.8441333333333333
```
And I didn't adjust any hyper parameters. After 15 epochs, the accuracy can reach 92.7%
```Phase val loss: 0.18356857439478239, acc: 0.9269333333333334```
## Dataset
For **classification** dataset: I use a [dataset](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) from kaggle which contains two classes(cats, dogs).
For detection dataset: In another [repo](https://github.com/ZhuYun97/ShufflNetv2-YOLOv3)
## References
- For structure: [shufflenet-v2](https://github.com/ericsun99/Shufflenet-v2-Pytorch)
- For data processing: [process data](https://zhuanlan.zhihu.com/p/29024978)

