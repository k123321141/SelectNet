## Dengue dataset introduction
總共有63項feature，其中有部分feature有缺失值，缺失處理則用平均值補上。</br>
目前人工找到用6個feature的表現最佳，分別是：</br>
WBC, Plt, Hb, age, Temp, sex</br>

## redundent feature
MAP = F(SBP, DBP)

## Experiment Assumption
1. feature selection improvement in accuracy performation.</br>

## Results

### Denoised improvement on missing value (overfit on noise feature)
首先是feature selection對於存在missing的資料集，能不能做到準確率上的改善。</br>
*假設存在missing value時，model可能透過fit training set裡面其他不相關的feature，而造成了overfitting。</br>
如果確實發生了overfitting，就可以看到train & val的差異。</br>*
![performance improvement][acc_train_vs_val]</br>
這張圖比較了train & val上的差異，在training set以及testing set都是有missing的。</br>
結論：v1的performance在train & val均低於v0, v2。而v0產生了overfitting，v2則保持相似的水準。</br>
由此看出v0受到missing noise的影響也許很大，但由於testing set其實是存在相同分布的missing，所以無法確認是否是missing造成，還是over-training。</br>

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

[synthesized_fig1]: https://github.com/k123321141/SelectNet/blob/master/figures/synthesized_fig1.png
[acc_train_vs_val]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/acc_train_vs_val.png
[acc_noised_train_vs_noised_val]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/acc_noised_train_vs_noised_val.png
[compare_ratio]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/compare_ratio.png
[reg_loss]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/reg_loss.png
[w_loss]: https://github.com/k123321141/SelectNet/blob/master/figures/results/synthesized/overall_0-2/w_loss.png

