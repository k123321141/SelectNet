# Original regression loss in Deep learning

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;f&space;\left&space;(&space;x&space;\mid&space;\Theta&space;\right&space;),&space;x\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;\|&space;\Theta&space;\right&space;\|_2&space;&plus;(\hat{y}&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;f&space;\left&space;(&space;x&space;\mid&space;\Theta&space;\right&space;),&space;x\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;\|&space;\Theta&space;\right&space;\|_2&space;&plus;(\hat{y}&space;-&space;y)^2" title="\hat{y} = f \left ( x \mid \Theta \right ), x\in R^d \newline Loss =\alpha\left \| \Theta \right \|_2 +(\hat{y} - y)^2" /></a></br></br>

左邊是L2 regularization tern，初步的想法是再增加一個限制項。</br>

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;=&space;f&space;\left&space;(&space;w&space;\odot&space;x&space;\mid&space;\Theta&space;\right&space;),&space;x,&space;w\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;\|&space;\Theta&space;\right&space;\|_2&space;&plus;&space;\beta\left&space;\|&space;w&space;\right&space;\|_1&plus;&space;(\hat{y}&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;=&space;f&space;\left&space;(&space;w&space;\odot&space;x&space;\mid&space;\Theta&space;\right&space;),&space;x,&space;w\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;\|&space;\Theta&space;\right&space;\|_2&space;&plus;&space;\beta\left&space;\|&space;w&space;\right&space;\|_1&plus;&space;(\hat{y}&space;-&space;y)^2" title="\hat{y} = f \left ( w \odot x \mid \Theta \right ), x, w\in R^d \newline Loss =\alpha\left \| \Theta \right \|_2 + \beta\left \| w \right \|_1+ (\hat{y} - y)^2" /></a></br></br>


透過增加L1 loss使得w收斂成比較sparse的向量，有些值為零。</br>
這時候可以得知哪些feature，是不需要使用的。</br>
詳細的計算式定義：</br>

<a href="https://www.codecogs.com/eqnedit.php?latex=w'&space;=&space;sigmoid(w)&space;\newline&space;w_{ratio}&space;=&space;w'&space;/&space;\sum^d_i&space;w_i',&space;w_{ratio}&space;\in&space;R^d&space;\newline&space;\hat{y}&space;=&space;f&space;\left&space;(&space;w_{ratio}&space;\odot&space;x&space;\mid&space;\Theta&space;\right&space;),&space;x,&space;w\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;\|&space;\Theta&space;\right&space;\|_2&space;&plus;&space;\beta\left&space;\|&space;w'&space;\right&space;\|_1&plus;&space;\gamma&space;entropy\left&space;(&space;w_{ratio}&space;\right&space;)&space;&plus;(\hat{y}&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w'&space;=&space;sigmoid(w)&space;\newline&space;w_{ratio}&space;=&space;w'&space;/&space;\sum^d_i&space;w_i',&space;w_{ratio}&space;\in&space;R^d&space;\newline&space;\hat{y}&space;=&space;f&space;\left&space;(&space;w_{ratio}&space;\odot&space;x&space;\mid&space;\Theta&space;\right&space;),&space;x,&space;w\in&space;R^d&space;\newline&space;Loss&space;=\alpha\left&space;\|&space;\Theta&space;\right&space;\|_2&space;&plus;&space;\beta\left&space;\|&space;w'&space;\right&space;\|_1&plus;&space;\gamma&space;entropy\left&space;(&space;w_{ratio}&space;\right&space;)&space;&plus;(\hat{y}&space;-&space;y)^2" title="w' = sigmoid(w) \newline w_{ratio} = w' / \sum^d_i w_i', w_{ratio} \in R^d \newline \hat{y} = f \left ( w_{ratio} \odot x \mid \Theta \right ), x, w\in R^d \newline Loss =\alpha\left \| \Theta \right \|_2 + \beta\left \| w' \right \|_1+ \gamma entropy\left ( w_{ratio} \right ) +(\hat{y} - y)^2" /></a></br></br>

目前正在研究加入entropy是否有幫助。</br>
透過對sigmoid(w)做L1 loss而不是直接針對w，使得w'可以持續下降到負數，相較於w只能下降到0更加smooth。</br>
並且就不需要使用softmax來模擬distribution，可以直接使用summation ratio，原因也是因為summation比較不陡峭，</br>
在w'極小值時，loss會比softmax指數還要大，藉此比較容易下降到0。</br>
目前只是實驗猜測，下列嘗試過的[Loss function](#other-loss)




## Toy example, bmi
y = w / (h^2) + small noise, scalar</br>
X是四維的向量，前兩維x1,x2 = w, h，後兩維x3,x4分別是不重要的normal distrbution。</br></br>
mse loss相當低代表network可以fit 這個回歸問題。</br>
w也持續下降中，而有幾項不相關的項，例如w3, w4下降的比較快</br></br>
![bmi w chart][bmi_w]</br></br>
最終ratio_w顯示，Network只使用了前兩項w,h就可以fit這個回歸問題，表示這個regularization tern可以找出哪些feature才是network有使用到的。</br>
w3, w4趨近於0</br></br>
![bmi ratio_w chart][bmi_w_ratio]</br></br>
summary</br></br>
![bmi log summary][bmi_summary]</br>



### 對於非相關項的噪音抵抗力



### 單變量統計


## other loss
[bmi_summary]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_summary.png
[bmi_w]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w.png
[bmi_w_ratio]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w_ratio.png
