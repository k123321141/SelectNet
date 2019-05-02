


### 與單變量統計以及線性模型的比較，XOR資料


### related papers
1. A Penalty-Function Approach for Pruning Feedforward Neural Networks, IEEE, 1997. 246 citations. </br></br>
short critique : 使用input-hidden-output共三層的FCN作為範例，對output node有影響的共有兩層learnable matrix, w1,w2</br>
只要其中某一條link(tanh(x*w1_i)*w2_j)的值足夠小便表示這條link可以刪除，w1_i & w2_j都可以被設為0。</br>
一樣透過regularization loss:</br> <a href="https://www.codecogs.com/eqnedit.php?latex=f(w)&space;=&space;\epsilon_1&space;\beta&space;w^2&space;/&space;(1&plus;\beta&space;w^2)&space;&plus;&space;\epsilon_2w^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(w)&space;=&space;\epsilon_1&space;\beta&space;w^2&space;/&space;(1&plus;\beta&space;w^2)&space;&plus;&space;\epsilon_2w^2" title="f(w) = \epsilon_1 \beta w^2 / (1+\beta w^2) + \epsilon_2w^2" /></a></br>
來限制weight。</br>
主要還是在討論如何做network pruning，這篇主要相關是下一篇同作者使用這種演算法來做feature selection。</br>
有提到一個L2 loss的變種，Loss = w^2 / (1 + w^2)，會將loss bound在[0, 1]，這可以加入regularization loss 設計討論。</br>


2. Neural-Network Feature Selector, IEEE, 1997. 427 citations.</br/br>
short critique : 承續上篇的loss，使得weight有sparsity的特性，討論如何做feature selection。</br>
Method : 首先使用全部N個feature訓練一個network，然後定義一個能夠接受的最小performance drop，在這篇裡面使用accuracy。</br>
再來分別抽掉一個feature，利用N-1個feature訓練出N-1個network，然後對performance做排序，只要最差的那個network還在允許performance drop範圍內，就繼續抽掉feature。</br>
具體來說是一個greedy strategy，有點像Decision Tree。</br>
Drawbacks : 首先是greedy的順序有沒有影響不確定，這是對於exhausive search for optimal solution的取捨。再者是訓練時間太過誇張。time complexity = O(N! - k!), k = final feature count.</br>

3. Feature selection with neural networks, 2001. 236 citaions.</br>
short critique : 
support : 裡面提到了有weights-based(上面那篇)跟signal-to-noise retio以及output sensitivity based以及三個方面，survey的時候可以注意。</br>

## other loss
[bmi_summary]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_summary.png
[bmi_w]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w.png
[bmi_w_ratio]: https://github.com/k123321141/SelectNet/blob/master/data/figures/bmi_w_ratio.png
