## harder dataset introduction
這是一個人工資料集，表示高維非線性的轉換，在x1, x2的分佈如下圖。</br>
![data distribution][harder_fig1]</br>
僅x1,x2為相關項，其餘一共增加了額外6項跟target function無關的變數。</br>
如果feature selection做得夠好，應該要從8個變數中，找出x1,x2才是有效變數。</br>


## redundent feature
新增一項-0.5x2，這是一個有效的feature，可以檢驗是否能夠濾除redundent feature。</br>

## random feature
新增兩項不同分布的random feature，分別是，uniform random跟normal random。</br>
其分佈涵蓋範圍跟x1,x2很相近，不是離群值。</br>

## random permutation
這是這篇的想法</br>
DeepPINK reproducible feature selection in deep neural networks, IEEE, 2018, 7</br>
作者針對每個feature單獨產生符合相應分佈但是卻完全與label獨立的假feature。</br>
我分別透過對x1, x1^2, x2 做random permutaion，如此一來便保證random permutaion是跟label無關，但保有某些性質。</br>
這邊也許可以再透過對不同feature group做random permutaion，以保證多變量的相關性(某種轉換下的同質feature group)</br>
x1 permutation, x1^2 permutation, x2 permutation</br>

## noise
2個有效feature加上6個應該被去除的feature總共8項，為了滿足論文假設1.feature selection會有改進</br>
我對8個feature都加入了5% missing noise，分別對8個feature獨立sample出各自5%當作missing，透過剩下95%計算出平均值取代missing value。</br>
如此一來model可能透過其他noise feature去overfit，如果能夠選出有效的feature就能避免overfit並改善準確度。</br>
值得一提的是當reduent feature沒有一起missing時，是否就能夠帶來好處而不被feature selection所拋棄？</br>

## Experiment Assumption
1. feature selection improvement in accuracy performation.</br>
2. 透過修改不重要的feature，可以看出模型對這些feature的敏感度，如此也能避免adversarial attack。</br>


## Results

### Denoised improvement on missing value (overfit on noise feature)
首先是feature selection對於存在missing的資料集，能不能做到準確率上的改善。</br>
*假設存在missing value時，model可能透過fit training set裡面其他不相關的feature，而造成了overfitting。</br>
如果確實發生了overfitting，就可以看到train & val的差異。</br>*
![performance improvement][acc_train_vs_val]</br>
這張圖比較了train & val上的差異，只有training set是有missing的。</br>
結論：唯v1的performance在train & val均低於v0, v2，並沒有看出overfit的問題，或是得到顯著improvement。</br>
也許是問題太困難，需要獨立用x1, x2來重新訓練一個模型，以確保DNN的能力無法處理這個問題，所以無法產生overfitting。</br>

### Denoised Capability
這邊要探討的是，當確定有些feature是不相關的，在不相關的feature上增加noise，正確的模型應該不受影響。</br>
*假設存在不相關feature時，正確的模型應該不受不相關的feature影響</br>
透過對不相關的feature加上random noise，如果模型學習錯誤的feature，表現就會降低。</br>*
![Denoised capability][acc_noised_train_vs_noised_val]</br>
這張圖比較了train & val上的差異，在訓練過程中，模型都沒有看過加過噪音的資料。</br>
結論：唯v1的performance在train & val均低於v0, v2，noise並沒有造成明顯影響，顯然三者模型都有denoised能力。</br>

### Feature Selection
這邊要討論如何根據模型收斂的結果，找出有效的feature。</br>
![Feature Selection][compare_ratio]</br>
相較v2的版本，v1很難看出來具體用了哪一些feature。</br>

### other drawbacks
L2 regularization loss由於v1會將input feature縮放到非常小，所以L2 regularization loss會相較的高些。</br>
![regularization loss][reg_loss]</br>
還有w loss收斂的情況(尚無結論)</br>
![w loss][w_loss]</br>


### 與network pruning的比較，CK的建議
[TensorFlow Model Optimization Toolkit — Pruning API][https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-pruning-api-42cac9157a6a?fbclid=IwAR2ZD9euHYiKh-OvuKYe4YltYCtosM_oJTlAdnN5AJwQUTDO9e8OW-8d9uM]
TODO




## norm 0.5
TODO

## model attack
針對多類別分類模型的防禦，可以嘗試對特定class做feature selection，然後把這個feature set當作filter，藉此達到feature squeeze的效果。</br>
TODO

[harder_fig1]: https://github.com/k123321141/SelectNet/blob/master/figures/harder_fig1.png
[acc_train_vs_val]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/acc_train_vs_val.png
[acc_noised_train_vs_noised_val]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/acc_noised_train_vs_noised_val.png
[compare_ratio]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/compare_ratio.png
[reg_loss]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/reg_loss.png
[w_loss]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/w_loss.png

