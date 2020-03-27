# Building extraction on satellite optic images
### A project done by Adrien Courtois and Kevin Mercier
#### Master MVA - Under the supervision of Florence Tupin and Emanuele Dalsasso

In this project, we trained an Attention U-Net [1] on the Inria dataset [2], and more especially on the city of Vienna. We also trained a 
model on the SpaceNet dataset [3], focusing on Paris.

## Performances
We reached an IOU of 78% on the Vienna dataset, of 76% on the Paris dataset RGB and of 79% on the Paris dataset 8 channels.

## Usage
### Evaluation
Once you've downloaded the weights, one can use one of the pretrained network using:

```python eval.py path-to-the-weights path-to-the-image```

The resulting mask will be saved as ```result.png``` in the current directory.

### Training
-- 

### Running a style transfer / color transfer algorithm
#### The Gatys algorithm
```python domaintransfer/gatys.py source.png target.png [-v|--verbose]```

The resulting image will be saved as ```gatys_result.png``` in the current directory.

#### The Reinhard color transfer algorithm
```python domaintransfer/reinhard.py source.png target.png```

The resulting image will be saved as ```reinhard_result.png``` in the current directory.

#### The color transfer algorithm based on Optimal Transport
```python domaintransfer/transfer.py source.png target.png```

The resulting image will be saved as ```transport_result.png``` in the current directory.

## Weights for the Vienna dataset
https://drive.google.com/open?id=1-eovkDaG2w_yOOUs4Ah2C8AsURm80nam

## Weights for the Paris dataset RGB
https://drive.google.com/open?id=1-baHHzfw1f6ohDPzcEfHlJPQ68LaLK4D

## Weights for the Paris dataset 8 channels
--

## References
[1] O. Oktay, J. Schlemper, L. Le Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Ham-merla, B. Kainz, B. Glocker, and D. Rueckert, “Attention u-net:  Learning where to look for the pancreas,”Arxiv:1804.03999

[2] E.  Maggiori,  Y.  Tarabalka,  G.  Charpiat,  and  P.  Alliez,  “Can  semantic  labeling  methods  generalize  to  anycity?  the inria aerial image labeling benchmark,”2017 IEEE International Geoscience and Remote SensingSymposium (IGARSS), Fort Worth, TX, 2017, pp. 3226-3229

[3] A.  Van  Etten,  D.  Lindenbaum,  and  T.  M.  Bacastow,  “Spacenet:   A  remote  sensing  dataset  and  challenge series,”Arxiv:1807.01232.
