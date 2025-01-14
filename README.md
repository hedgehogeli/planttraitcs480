# planttraitcs480
![alt text](https://github.com/hedgehogeli/planttraitcs480/blob/main/basic.png?raw=true)

Given an image of a plant and corresponding ancillary data pulled from open databases, predict plant features. 

Basic architecture processes image and ancillary data separately, then concatenates into linear head. 

Pretrained ViTs (e.g. dino) and CNNs (e.g. VGGnet), as well as untrained ViTs and CNNs were cross-compared. 

Transformers were attempted on the ancillary data, but rapid overtraining was observed. 

Hyperparameters and size of linear layers were also tuned somewhat. 

![alt text](https://github.com/hedgehogeli/planttraitcs480/blob/main/1.png?raw=true)
![alt text](https://github.com/hedgehogeli/planttraitcs480/blob/main/2.png?raw=true)
![alt text](https://github.com/hedgehogeli/planttraitcs480/blob/main/5.png?raw=true)

![alt text](https://github.com/hedgehogeli/planttraitcs480/blob/main/r2.png?raw=true)
![alt text](https://github.com/hedgehogeli/planttraitcs480/blob/main/loss.png?raw=true)
