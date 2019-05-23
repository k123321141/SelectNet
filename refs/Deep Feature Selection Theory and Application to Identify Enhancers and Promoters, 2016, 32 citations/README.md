# Deep Feature Selection Theory and Application to Identify Enhancers and Promoters, 2016, 32 citations
這篇的概念是從Lasso norm1 regularization loss 延伸而來。</br>
首先有一個比較簡單的變形叫做elastic net，嘗試做norm1 跟 norm2 之間的平衡。</br>
再來延伸到neural network。</br>

直接看模型圖</br>
![model architecture][fig1]</br>
這是跟lasso的延伸。</br>
![lasso][exp1]</br>
透過對單層的w1做norm1，期待收斂到一個sparse的解，用以做feature selection。</br>
再加上後面network的l2 norm，避免在這一層縮小了feature的影響程度，在後面又被極大的weight給放大回來。</br>

## Data
這篇用的資料也是醫學相關，是DNA的資料集。</br>
我找不到線上可以用的資源，從描述上來看是有人工標注了哪些feature並不重要</br>
總共93個feature，然後做分三類的分類問題。</br>

## Result
![result][fig2]</br>

跟各個feature在不同次模型裡，出現次數的heatmaps。</br>
![heatmaps][fig3]</br>


[fig1]: https://github.com/k123321141/SelectNet/blob/master/refs/Deep%20Feature%20Selection%20Theory%20and%20Application%20to%20Identify%20Enhancers%20and%20Promoters%2C%202016%2C%2032%20citations/fig1.png
[fig2]: https://github.com/k123321141/SelectNet/blob/master/refs/Deep%20Feature%20Selection%20Theory%20and%20Application%20to%20Identify%20Enhancers%20and%20Promoters%2C%202016%2C%2032%20citations/fig2.png
[fig3]: https://github.com/k123321141/SelectNet/blob/master/refs/Deep%20Feature%20Selection%20Theory%20and%20Application%20to%20Identify%20Enhancers%20and%20Promoters%2C%202016%2C%2032%20citations/fig3.png
[exp1]: https://github.com/k123321141/SelectNet/blob/master/refs/Deep%20Feature%20Selection%20Theory%20and%20Application%20to%20Identify%20Enhancers%20and%20Promoters%2C%202016%2C%2032%20citations/exp1.png
