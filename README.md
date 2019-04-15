### Original regression loss in Deep learning

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;f&space;(&space;x&space;|&space;\Theta),&space;x\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;|&space;\Theta^2\right&space;|&space;&plus;(\hat{y}&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;f&space;(&space;x&space;|&space;\Theta),&space;x\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;|&space;\Theta^2\right&space;|&space;&plus;(\hat{y}&space;-&space;y)^2" title="\hat{y} = f ( x | \Theta), x\in R^d \newline Loss =\alpha\left | \Theta^2\right | +(\hat{y} - y)^2" /></a></br></br>

左邊是L2 regularization tern，初步的想法是再增加一個限制項。</br>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{y}&space;=&space;f&space;(&space;w\odot&space;x&space;|&space;\Theta),&space;x,w&space;\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;|&space;\Theta^2&space;\right&space;|&space;&plus;&space;\beta\left&space;|&space;w&space;\right&space;|&space;&plus;(\hat{y}&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y}&space;=&space;f&space;(&space;w\odot&space;x&space;|&space;\Theta),&space;x,w&space;\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;|&space;\Theta^2&space;\right&space;|&space;&plus;&space;\beta\left&space;|&space;w&space;\right&space;|&space;&plus;(\hat{y}&space;-&space;y)^2" title="\hat{y} = f ( w\odot x | \Theta), x,w \in R^d \newline Loss =\alpha\left | \Theta^2 \right | + \beta\left | w \right | +(\hat{y} - y)^2" /></a></br>

透過增加L1 loss使得w收斂成比較sparse的向量，有些值為零。</br>
這時候可以得知哪些feature，是不需要使用的。</br>
詳細的計算式定義：</br>

<a href="https://www.codecogs.com/eqnedit.php?latex=att\_w&space;=&space;sigmoid(x\odot&space;w)&space;\newline&space;att\_ratio&space;=&space;att\_w/&space;\sum{att\_w}&space;\newline&space;\hat{y}&space;=&space;f&space;(&space;att\_ratio\odot&space;x&space;|&space;\Theta),&space;x,w&space;\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;|&space;\Theta^2&space;\right&space;|&space;&plus;&space;\beta\left&space;|&space;att\_w&space;\right&space;|&space;&plus;(\hat{y}&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?att\_w&space;=&space;sigmoid(x\odot&space;w)&space;\newline&space;att\_ratio&space;=&space;att\_w/&space;\sum{att\_w}&space;\newline&space;\hat{y}&space;=&space;f&space;(&space;att\_ratio\odot&space;x&space;|&space;\Theta),&space;x,w&space;\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;|&space;\Theta^2&space;\right&space;|&space;&plus;&space;\beta\left&space;|&space;att\_w&space;\right&space;|&space;&plus;(\hat{y}&space;-&space;y)^2" title="att\_w = sigmoid(x\odot w) \newline att\_ratio = att\_w/ \sum{att\_w} \newline \hat{y} = f ( att\_ratio\odot x | \Theta), x,w \in R^d \newline Loss =\alpha\left | \Theta^2 \right | + \beta\left | att\_w \right | +(\hat{y} - y)^2" /></a></br>

透過對sigmoid(w)做L1 loss而不是直接針對w，使得att_w可以持續下降到負數，相較於w只能下降到0更加smooth。</br>
並且就不需要使用softmax來模擬distribution，可以直接使用summation ratio。</br>
目前只是實驗猜測，下列嘗試過的Loss function(#other-loss)




### Toy example, bmi
y = w / (h^2) + small noise, scalar</br>
X是四維的向量，前兩維x1,x2 = w, h，後兩維x3,x4分別是不重要的normal distrbution。</br>
![Alt text][bmi_summary]


### 對於非相關項的噪音抵抗力



### 單變量統計




[other loss function](#other-loss)

[bmi_summary]: https://github.com/k123321141/SelectNet/data/figures/bmi_summary.png
