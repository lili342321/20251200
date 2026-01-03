# Fractional-Order Fuzzy Cognitive Maps: A Framework for Long-Term Forecasting of Multivariate Time Series
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)



<p align="center">
<img src=".\pic\FFCM.png" height = "360"   alt="" align=center />  
<br><br>
</p>

1.


##  1.Program Overview
(1) First, multidimensional Gaussian information granules are used to granulate the data in the training set, thereby obtaining the optimal segmentation method. For specific details, please refer to the literature “_Design Gaussian information granule based on the principle of justifiable granularity: A multi-dimensional perspective._” (Since this part of the method is adapted from existing work, I have reproduced the approach accordingly. The code is not publicly available at present, but can be provided upon request.)

(2) Next, run the “main_fcm” file in the util folder. Commands for training *VI-DFCM*  on the SIEC dataset:
  ```bash
   # SIEC
  --data ./steel --grain_seq_len 32 --grain_step_start 24 --grain_step 8
  ```

The key parameters are introduced as follows:
  `grain_seq_len` represents the length of each information granule, `grain_step_start` denotes the non-overlapping length of each granule and also serves as the prediction length for each iteration, while `grain_step` is the overlapping   length between neighboring granules. The values of `grain_seq_len` and `grain_step` are obtained from the segmentation results in the first step.

  Among these parameters, `grain_step_start` is a hyperparameter, and you only need to select an appropriate value based on the input and output length requirements of the task. For example, if the maximum input sequence length is 50 and the output length is 168, then 24 is chosen for two reasons: (1) this approach provides sufficient space for searching the optimal sliding window length, and the information granule length can be optimized within the range of 25 to 50; (2) the output length of 168 is an integer multiple of 24.




## 2.Paper Overview



Figure 1 shows the overall training process of VI-DFCM. VI-DFCM achieves long-term prediction through the synergistic effect of multiple iterations by the VI-FCM submodel and a single iteration by the Cross VI-FCM submodel.

### (1).Data Preprocessing



In our designed experiments, the prediction lengths are {168, 720}, and the input sequence lengths for each baseline model are {50, 100}, respectively. Therefore, when performing information granulation, we require that the length of the obtained optimal information granule does not exceed the input sequence length of each baseline, i.e., the input length of VI-DFCM should be less than or equal to {50, 100}.
### (2).VI-FCM submodel


The inference process of the VI-FCM submodel. The transfer function of VI-FCM is given by:

$A_i(S+1) = \frac{1}{1 + e^{-\lambda \left( \sum_{j=1}^{N} \left(\gamma_{i,j}  w_{i,j,:} \right) \circ A_j(S) + u_i(S) \right)}}$

where $\circ$ denotes the Hadamard product, $\lambda$ represents the steepness parameter of the transfer function near zero, and $\boldsymbol{w}\in\mathbb{R}^{N\times N \times I}$ is the weight tensor. The weights $\boldsymbol{w}$, modulated by the dynamic coefficients $\gamma_{i,j}\in\mathbb{R}$, are then element-wise multiplied with the node  $\boldsymbol{A}$ via the Hadamard product. In the transfer function, each scalar $\gamma_{i,j}$ multiplies its corresponding weight vector $\boldsymbol{w_{ij:}\in\mathbb{R}^{I}}$. As shown in Figure 1, the full coefficient matrix $\gamma\in\mathbb{R}^{N\times N}$ and the weight tesnor $\boldsymbol{w}$ also undergo element-wise multiplication. Although these expressions differ in form, they convey the same meaning.

🔎 Equations `(12)–(15)` correspond to code in `models/VIDFCM.py`, lines `25–54`. The code on line `39` represents the Hadamard product of the weights $\boldsymbol{w}$ and the node matrix $\boldsymbol{A}$. 
### (3).Cross VI-FCM submodel


The inference process of the Cross VI-FCM submodel. The transfer function of Cross VI-FCM is given by:

$A_i(S+1) = \frac{1}{1 + e^{-\lambda \left( \sum_{j=1}^{N}  \left(\gamma_{i,j}^{CF}  w_{i,j,:}\right) \circ A_j(S) + u_i(S) \right)}}$

Similarly to the VI-FCM submodel, the weights $\boldsymbol{w}$, modulated by the dynamic coefficients $\gamma_{i,j}$, are then element-wise multiplied with the node  $\boldsymbol{A}$ via the Hadamard product. The code on line 76 represents the Hadamard product of the weights $\boldsymbol{w}$ and the node matrix $\boldsymbol{A}$. 

🔎 Equation `(18)` correspond to code in `models/VIFCM_Cross.py`, lines `31`.

🔎 Equations `(19)–(22)` correspond to code in `models/VIDFCM.py`, lines `68–91`.




## 3.Requirements

- Python 3.9
- numpy ==  1.26.3
- pandas == 2.2.3
- torch ==  2.6.0
- tqdm == 4.67.1




