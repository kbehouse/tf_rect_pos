## Use tensorflow to predict the rectangle position 

This repo accomplish the following idea, predict the rect image(pixel) position 

[![Predict rect posistion](https://img.youtube.com/vi/HuNF8_PzlXI/0.jpg)](https://www.youtube.com/watch?v=HuNF8_PzlXI)

## Generate big data and normalize

Generate big data images in 10 batchs(train_data_[00~09]/), each batch has 30,000 images and one _pos.out record rectangle position 

```bash
python3 gen_rect_patterns.py
```

Normalize all images  from 0~255 to 0~1 and save to pickle (batch_[00~09].p)

```bash
python3 preprocess_normalize_and_save_pickle.py
```

## Train model and test model

Train and save model to checkpoints
Default prefer_loss <= 1.5

```bash
python3 train_rect_pos.py
```

Test the model, you could move mouse ***Cursor*** in the window, and save to valid_pic/
```bash
python3 validation_rect_pos.py
```

## Acknowledgements
[GUI origin by MorvanZhou](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/experiments/Robot_arm)


## License
MIT
