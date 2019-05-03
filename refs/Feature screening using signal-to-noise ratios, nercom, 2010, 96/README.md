


# Feature screening using signal-to-noise ratios, nercom, 2010. 96 citations.
short critique : 這篇沒什麼作用，主要是提出的方法並不實用，但是提供了另一種衡量feature saliency 的方式，SNR based。</br></br>
support : 裡面分了三個feature saliency measure. Partial derivative based, weighted based and SNR based.</br>


## content
裡面採用的例子是single layer fully-connected network，如果要推廣到其他架構的話需要額外補充，現就這個簡單例子說明。</br>
Partial derivative based : 直接計算對xi的偏微分值，以gradient表示該feature的重要性，要注意是activation function需要被計算。</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\Lambda_i&space;=&space;\frac{1}{K}&space;\cdot&space;\frac{1}{M_{train}}\cdot\sum^K_{k=1}\sum^{M_{train}}_{m=1}\left&space;|&space;\frac{\partial&space;z_{k,m(x_m,W)}}{\partial&space;x_{i,m}}&space;\right&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Lambda_i&space;=&space;\frac{1}{K}&space;\cdot&space;\frac{1}{M_{train}}\cdot\sum^K_{k=1}\sum^{M_{train}}_{m=1}\left&space;|&space;\frac{\partial&space;z_{k,m(x_m,W)}}{\partial&space;x_{i,m}}&space;\right&space;|" title="\Lambda_i = \frac{1}{K} \cdot \frac{1}{M_{train}}\cdot\sum^K_{k=1}\sum^{M_{train}}_{m=1}\left | \frac{\partial z_{k,m(x_m,W)}}{\partial x_{i,m}} \right |" /></a></br>
K是feature set，M是training set。</br>

weight-based : 計算所有第一層跟feature xi相關weight的L2 norm，稱之為Tarr's silency measure。</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\tau_i&space;=&space;\sum^J_{j=1}(w^1_{i,j})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_i&space;=&space;\sum^J_{j=1}(w^1_{i,j})^2" title="\tau_i = \sum^J_{j=1}(w^1_{i,j})^2" /></a></br>
i表示第幾個feature，J表示第一層的wieght set，這邊有個問題是CNN, RNN等等有關weight sharing的部分不好延伸，因為也許過完activation function其作用為0。</br>
論述是，當Tarr's silency measure很低時，則這個feature不重要。</br>

SNR based : 透過新增一個與output完全無關的noise input(比例中使用uniform [0,1])，加入input layer，命名為noise node，其相關的weight應該具備某種weight-based的特性，再透過比對正常feature與noise feature的相似度(差異)，就可以說明有同樣特性的feature很可能跟noise feature一樣，不影響outcome。(各feature已經經過normalization)。</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=SNR_i&space;=&space;10\log_{10}(\frac{\sum^J_{j=1}(w^1_{i,j})^2}{\sum^J_{j=1}(w^1_{N,j})^2)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SNR_i&space;=&space;10\log_{10}(\frac{\sum^J_{j=1}(w^1_{i,j})^2}{\sum^J_{j=1}(w^1_{N,j})^2)})" title="SNR_i = 10\log_{10}(\frac{\sum^J_{j=1}(w^1_{i,j})^2}{\sum^J_{j=1}(w^1_{N,j})^2)})" /></a></br>
wi,j一樣表示第一層weight對於第i個feature對第j個neuron的weight，而N則表示noise feature。</br>
我不太清楚這算是什麼轉換，但是文中說明這是一種轉到分貝的轉換(?)。</br>
當不重要的feature加總的值與noise feature相似時，log 1=0，其SNR會在零附近擺盪。</br>
而重要的feature由於加總會大於零，增長為某個大數之後經過對數轉換，則會趨近於某個大於零的常數。</br>
由此SNR可以做排序。</br>
在這順便說明他的實驗設計，透過27個不同的模型參數，(3種hidden dimension，3種learning rate，3種momentum rate)</br>
每種各訓練10次，總共270次實驗，做平均後去觀察每個feature的重要排序程度。</br>
原先的feature有4種，再加上額外4個不相干的noise feature (Uniform[0,1])，然後再加上noise node跟bias。總共10個。</br>
每次記錄一下每個feature的排名，圖中不包含noise node跟bias的排名。</br>
黑線代表noise node的排名。</br>
![Feature Ranking][fig1]</br>
可以看得出來noise feature 跟 noise node的分布很相似，但是作者還是利用了相關性統計檢定來說服讀者。</br>
spearman's correlation test</br>
再來則是描述做feature seleciton的作法：
1. 訓練直到SNR穩定，記錄在test set上的performance。
2. 拿掉最低SNR的feature。
接下來兩張圖則是用本文中提到的做法，逐次拿掉feature造成的performance drop。</br>
以及PCA投影的performance drop。</br>
![Feature Ranking][fig2]</br>
![Feature Ranking][fig3]</br>
作者想說明的是，文中提到的方法與PCA的效果相似，所以是合理的。</br>

[fig1]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20screening%20using%20signal-to-noise%20ratios%2C%20nercom%2C%202010%2C%2096/fig1.png
[fig2]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20screening%20using%20signal-to-noise%20ratios%2C%20nercom%2C%202010%2C%2096/fig2.png
[fig3]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20screening%20using%20signal-to-noise%20ratios%2C%20nercom%2C%202010%2C%2096/fig3.png
