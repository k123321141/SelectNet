
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

## MNIST example
由於numerical data比較難以找出feature saliency。</br>
所以先使用圖片




### 對於非相關項的噪音抵抗力



### 單變量統計


## other loss
[bmi_summary]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_summary.png
[bmi_w]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w.png
[bmi_w_ratio]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w_ratio.png


### 與單變量統計以及線性模型的比較，XOR資料

### feature ranking
有沒有可能每次用不同組的feature去做過濾，然後留下來的feature就得分。</br>
以得分高低來說明feature的重要性。</br>

### related papers
1. A Penalty-Function Approach for Pruning Feedforward Neural Networks, IEEE, 1997. 246 citations. </br></br>
short critique : 使用input-hidden-output共三層的FCN作為範例，對output node有影響的共有兩層learnable matrix, w1,w2</br>
只要其中某一條link(tanh(x*w1_i)*w2_j)的值足夠小便表示這條link可以刪除，w1_i & w2_j都可以被設為0。</br>
一樣透過regularization loss:</br> <a href="https://www.codecogs.com/eqnedit.php?latex=f(w)&space;=&space;\epsilon_1&space;\beta&space;w^2&space;/&space;(1&plus;\beta&space;w^2)&space;&plus;&space;\epsilon_2w^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(w)&space;=&space;\epsilon_1&space;\beta&space;w^2&space;/&space;(1&plus;\beta&space;w^2)&space;&plus;&space;\epsilon_2w^2" title="f(w) = \epsilon_1 \beta w^2 / (1+\beta w^2) + \epsilon_2w^2" /></a></br>
來限制weight。</br>
主要還是在討論如何做network pruning，這篇主要相關是下一篇同作者使用這種演算法來做feature selection。</br>
有提到一個L2 loss的變種，Loss = w^2 / (1 + w^2)，會將loss bound在[0, 1]，這可以加入regularization loss 設計討論。</br>


2. Neural-Network Feature Selector, IEEE, 1997. 427 citations.</br></br>
short critique : 承續上篇的loss，使得weight有sparsity的特性，討論如何做feature selection。</br>
Method : 首先使用全部N個feature訓練一個network，然後定義一個能夠接受的最小performance drop，在這篇裡面使用accuracy。</br>
再來分別抽掉一個feature，利用N-1個feature訓練出N-1個network，然後對performance做排序，只要最差的那個network還在允許performance drop範圍內，就繼續抽掉feature。</br>
具體來說是一個greedy strategy，有點像Decision Tree。</br>
Drawbacks : 首先是greedy的順序有沒有影響不確定，這是對於exhausive search for optimal solution的取捨。再者是訓練時間太過誇張。time complexity = O(N! - k!), k = final feature count.</br>

3. Feature screening using signal-to-noise ratios, nercom, 2010. 96 citations.</br></br>
short critique : 這篇沒什麼作用，主要是提出的方法並不實用，但是提供了另一種衡量feature saliency 的方式，SNR based。</br></br>
support : 裡面分了三個feature saliency measure. Partial derivative based, weighted based and SNR based.</br>

## other loss
[bmi_summary]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_summary.png
[bmi_w]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w.png
[bmi_w_ratio]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w_ratio.png


4. Feature selection with neural networks, 2001. 236 citaions.</br>
short critique : 
support : 裡面提到了有weights-based(上面那篇)跟signal-to-noise retio以及output sensitivity based以及三個方面，survey的時候可以注意。</br>

5. Feature Selection Based on Structured Sparsity- A Comprehensive Study, IEEE, 2017. 95</br>
在考慮不同feature group，要實現feature ranking時，可以考慮group lasso。</br>
注意一下matrix norm。 norm2,1會造成出現zero row而norm1,2 出現zero column，</br>
這代表要出現dead neuron，對所有connected link的weight都是零。</br>
或是filter neuron，只對特定幾個connected link的weight是零。</br>

