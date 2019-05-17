# Feature Selection- A Data Perspective, IEEE, 2018, 265 citations.
最重要的是我發現其中有一篇跟我的題目一模一樣，我難過。</br>
[Deep Feature Selection Theory and Application to Identify Enhancers and Promoters, 2016][1]
這篇提供非常多不同類別的方法，我打算照文章順序描述這篇。</br>
根據作者分類共分成</br>
1. [Similarity based](#similarity-based)
2. [Infomation Theoretical based](#infomation-theoretical-based)
3. [Sparse Learning based](#sparse-learning-based)
4. [Statistical based](#statistical-based)
5. [Other](#other) <- 本篇重點

首先用一張圖表示Feature Selection帶來的好處。</br>
![Feature Ranking][fig1]</br>
接下來我會節錄跟我的題目有關的部分，也就是跟supervise learning跟一點點autoencoder的部分。</br>
有些領域我會全部跳過，例如semi-supervised跟streaming data。</br>


## Similarity based

作者總結了一個廣泛的表示式，去理解每個方法其實就是這個表示式的特殊解。</br>
這邊直接節錄文章。</br>
![exp][exp1]</br>
簡單解釋一下再說明什麼，S是原先N個sample之間兩兩相對的距離，U(f)表示了某種利用sub feature的transform。</br>
這表達式說明了，找出一種transform能夠保留各個sample之間的距離與原先的feature set仍保留最大的相似度。</br>
這邊對我幫助不大就跳到作者的結論，這種找相似度的方法並不限定在supervised learning上，但是無法處理feature redundance。</br>

## Infomation Theoretical based

這作法延伸出來其實就是Decision Tree，所以是單變量information gain的排序。</br>
需要注意的是這種方法只能做在離散的資料上。</br>
首先是第一個最簡單的，information gain</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=J_{MIM}X_k=I(X_k;Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J_{MIM}X_k=I(X_k;Y)" title="J_{MIM}X_k=I(X_k;Y)" /></a></br>
這其實就是對每個feature算一次information gain然後排大小。</br>

第二種</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=J_{MIFS}X_k=I(X_k;Y)-\beta&space;\sum_{X_j\in&space;S}I(X_k;X_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J_{MIFS}X_k=I(X_k;Y)-\beta&space;\sum_{X_j\in&space;S}I(X_k;X_j)" title="J_{MIFS}X_k=I(X_k;Y)-\beta \sum_{X_j\in S}I(X_k;X_j)" /></a></br>
想要處理feature redundance，所以X k除了要跟label高度相關以外，也要跟其他feature的相關性足夠低才行。</br>
引入一個跟其他feature相關性的懲罰項來處理這個問題。</br>

第三種</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=J_{CMI}X_k=I(X_k;Y)-\beta&space;\sum_{X_j\in&space;S}I(X_k;X_j)&plus;\lambda&space;\sum_{X_j\in&space;S}I(X_j;X_k\mid&space;Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J_{CMI}X_k=I(X_k;Y)-\beta&space;\sum_{X_j\in&space;S}I(X_k;X_j)&plus;\lambda&space;\sum_{X_j\in&space;S}I(X_j;X_k\mid&space;Y)" title="J_{CMI}X_k=I(X_k;Y)-\beta \sum_{X_j\in S}I(X_k;X_j)+\lambda \sum_{X_j\in S}I(X_j;X_k\mid Y)" /></a>
希望結合多個feature以後對分類有幫助，所以又加入了一個conditional redundancy，如果加入這個feature對於分類有幫助。</br>
那麼就應該被加分。</br>
作者的結論是，這方法很快，而且可以處理feature redundancy，但是只能坐在離散資料上，以及supervised data。</br>
我認爲這方法比較多人用的方式是去看random forest的feature出現情況。</br>

## Sparse Learning based

這篇其實是源自於Lasso以及其延伸推廣，直接到結論。</br>
好處是直接加入learning的方法裡，通常會有很好的performance。</br>
但是通常牽涉到一些求解困難的問題，non-smooth optimization problem。</br>

## Statistical based

這沒什麼好講的，就是一些常見的統計分析。</br>
Gini-index, T-test, Chi-square test還有一些相關性統計。</br>
缺點是沒辦法做排序，以及處理多變量的缺憾。</br>

## Other

接下來就是一堆新秀崛起，還沒有被視為「傳統」的做法，</br>
第一個就是跟我衝題目的部分，幹。</br>
[Deep Feature Selection Theory and Application to Identify Enhancers and Promoters, 2016][1]</br>

第二篇</br>
Feature Selection using Deep Neural Networks, 2017</br>
針對relu，計算第一層有被activate cell的相關weight，透過統計sample裡面平均的weight說明這個feaure的重要程度。</br>

第三篇</br>
DeepPINK reproducible feature selection in deep neural networks, IEEE, 2018, 7</br>
用一張圖講解他，作者針對每個feature單獨產生符合相應分佈但是卻完全與label獨立的假feature。</br>
然後讓network去選，低weight的表示這個feature可能不重要。</br>
![DeepPINK architecture][fig4]</br>


[1]: https://github.com/k123321141/SelectNet/blob/master/refs/Deep%20Feature%20Selection%20Theory%20and%20Application%20to%20Identify%20Enhancers%20and%20Promoters%2C%202016%2C%2032%20citations/README.md
[fig1]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20Selection-%20A%20Data%20Perspective%2C%20IEEE%2C%202018%2C%20265/fig1.png 
[fig2]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20Selection-%20A%20Data%20Perspective%2C%20IEEE%2C%202018%2C%20265/fig2.png
[fig3]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20Selection-%20A%20Data%20Perspective%2C%20IEEE%2C%202018%2C%20265/fig3.png
[fig4]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20Selection-%20A%20Data%20Perspective%2C%20IEEE%2C%202018%2C%20265/fig4.png
[exp1]: https://github.com/k123321141/SelectNet/blob/master/refs/Feature%20Selection-%20A%20Data%20Perspective%2C%20IEEE%2C%202018%2C%20265/exp1.png
