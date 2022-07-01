# OCR

## Overview
Create sequence of digits as image using MINST dataset for OCR
- Length ranges from 1 to 10
- Overlay image on a white background of size 300 by 30
- Train with CRNN models
- Train using CTC Loss

## Run
- Setup with conda
```
conda create env -f environment.yml
```
- Train
```
python3 main.py --cfg cfgs/text_detection_rec_crnn.yml
```
- Test (Change the ckpt to the correct model)
```
python3 main.py --cfg cfgs/text_detection_rec_crnn.yml --mode test
```
- Sample logs of training (Epoch 7 has 96.84%)
```
Epoch: 1/300, Train Loss=0.906320323, Val Loss=0.0485067945, LR=0.0003
Epoch: 2/300, Train Loss=0.0359529055, Val Loss=0.0348301762, LR=0.0003
Epoch: 3/300, Train Loss=0.0240776808, Val Loss=0.0349971408, LR=0.0003
Epoch: 4/300, Train Loss=0.0168070863, Val Loss=0.0245943149, LR=0.00027
Epoch: 5/300, Train Loss=0.0142404981, Val Loss=0.0312365647, LR=0.00027
Epoch: 6/300, Train Loss=0.0112481162, Val Loss=0.0241795337, LR=0.00027
Epoch: 7/300, Train Loss=0.0090863228, Val Loss=0.0195328601, LR=0.000243
Epoch: 8/300, Train Loss=0.0077383744, Val Loss=0.0224177841, LR=0.000243
Epoch: 9/300, Train Loss=0.0065656391, Val Loss=0.0287263323, LR=0.000243
```