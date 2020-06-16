=================================DATA1=================================
information: 只要有其中一個landmark是被cut-off則被過濾
filename: train_list.txt

*IMAGE
-train data: 9,104
-valid data: 2,276

* LANDMARKS
-number_visible: 54,255
-number_occluded: 14,025

class: 18([visible/x/y]，6類，["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"])

=================================DATA2=================================
information: 全部
filename: train_list1.txt

*IMAGE
train data: 26,900
valid data: 6,724

* LANDMARKS
-number_visible: 138,671
-number_occluded: 47,073
-number_invisible: 16,000

class: 18([visible/x/y]，6類，["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"])

---------------------------------------------------------------------------------------------------------------------------------
[version 1]

*Pre-processing:
resize: 224*224

*Model:
input: 224*224*3
ouput: (12), (6)			# 6類的(x, y)座標, 6類的visible
Architecture: DenseNet121

*training:
optimizer: Adam
loss: Deepfashion1 loss

---------------------------------------------------------------------------------------------------------------------------------
[version 2]

* 資料預處理:
-image: resize(224*224)
-label: 1. heatmap(在黑mask上，以landmark上下左右各擴 5 pixel 變成一個11*11的白點)，再resize成224*224
	2. visible由 0 and 1 構成的tensor
* 模型架構:
input: 224*224*3
output: relu(224*224*6), sigmoid(6)	# 6類的heatmap(大小為224*224), 6類的visible
Architecture: HRNetFashionNet()
* 訓練參數:
batch_size = 16
optimizer: Adam(lr=0.0001)
loss: criterionHeat = MSELoss()
* 結論：學壞，不知道是不是因為激發函數用relu的關係所以嘗試用用看sigmoid(v3)

---------------------------------------------------------------------------------------------------------------------------------
[version 3]

*跟version2只改了output: sigmoid(224*224*6), sigmoid(6)
*結論：predict heatmap 最大值不只一個，有很多個，所以學壞原因應該不是因為激發函數，猜測應該是因為自己拉的HRNet層數太少？所以試試看用少層一點的DenseNet試試看(v4)

---------------------------------------------------------------------------------------------------------------------------------
[version 4]

*跟version3只改了Architecture: DenseNet121Heat()
*結論：結果正常一點，但感覺還是有些點的結果很差，猜測是學習次數不夠多的原因

---------------------------------------------------------------------------------------------------------------------------------
[version 5]

*跟version4只改了 
1.lr=0.001
2.batch_size=32
*結論：使用較大的lr+reduceLearningRate，結果正常了，某些被遮擋的點結果較不好

---------------------------------------------------------------------------------------------------------------------------------
[version 6]heat

*發現之前 heatmap gt 處理有問題，cut-off的點應該為全黑的 heatmap
*結論：效果比[v5]好一點

---------------------------------------------------------------------------------------------------------------------------------
[version 7]

*加入另一個loss： vis_criterion = nn.L1Loss()
*結論：爛！無貢獻！但必須思考如何訓練「點的可見與否」

---------------------------------------------------------------------------------------------------------------------------------
[version 8]bbox (**epoch100**)
* test_heat
* 在version 6 之上，加入衣服的bbox縮小landmarks 預測的範圍，想法為，若是將此系統套用到虛擬試衣上，過程會對衣服做segmentation，等於可以得到衣服的bbox，因此不算多作工
* 修改：
* 資料預處理:
- image: 對衣服ROI部份做crop再resize成224*224
- label: landmark 為了對應衣服ROI因此新的landmark為ROI的相對座標，加入bbox為了給原圖crop
* TESTING
* 資料後處理: 對 predict heatmap 做高斯濾波，但不使用後處理高斯濾波結果反而比較好=>可能heatmap_gt為高斯比較好(v18)
* 結論：使用衣服ROI使預測結果變好，但還是有部份遮擋部份預測不佳的問題，且因為heatmap based method無法學習全局點&點之間關聯，所以參考paper「Semantic Alignment」加入GHCU(version 9)

---------------------------------------------------------------------------------------------------------------------------------
[version 9] 加入GHCU
1. 使用 v8 的 model predict heatmap (stage1)
2. 將 heatmap 作為input 輸入GHCU預測6個landmarks的座標，去學點跟點之間的關聯(stage2)
3. 使用MSELoss()計算六個點的(x, y)差異
* 結論：因為考慮點跟點之間的關聯，因此結果更為穩定，但還是有少少數長袖（袖子部份）預測結果會偏移

---------------------------------------------------------------------------------------------------------------------------------
[version 10] GHCU改loss (加入visible 預測）
* Loss Function： criterion_GHCU()
* 結論：和v9 結果差不多，但似乎有些袖子的點預測比較準確，out_vis的值看起來都差不多，沒有什麼根據
* Evaluate: v8-100 + v10-70 Average NMSE = 0.011279039175802549

---------------------------------------------------------------------------------------------------------------------------------
[version 11] 其他嘗試
1. 在GHCU 的 input 加入原圖tensor當參考
* 結論：和v10 結果差不多
2. 在stage1之後，直接把heatmap當作input預測六個點的visible
*結論：不好，因為沒有座標資訊，model無從學習landmark是否可見
3. 

---------------------------------------------------------------------------------------------------------------------------------
[version 12]參考 Yolo loss 訓練 Landmark Visible
* 資料預處理:
-image: 取衣服 ROI resize(224, 224)
-label: 1. X 座標做 normalize 至 0-1
	2. Y 座標做 normalize 至 0-1
	3. confidence_nocut [0 or 1]
	4. confidence_vis   [0 or 1]
-Landmark類型:  1. cut-off	(Cnocut, Cvis) = (0, 0)
		2. occlusion	(Cnocut, Cvis) = (1, 0)
		3. visible	(Cnocut, Cvis) = (1, 1)
* 模型架構:
-input: 224*224*3
-output: LeakyReLU(12), sigmoid(6), sigmoid(6)	# 六的點的 (X, Y, Cnocut, Cvis)
-Architecture: LVNet()
* 訓練參數:
-batch_size=64
-LambdaCOOR = 1
-LambdaNOCUT = 0.07
-LambdaCUT = 0.83
-LambdaVIS = 0.1
* LOSS FUNCTION
-function: criterionLV()
-loss = LambdaNOCUT*loss_coord + LambdaNOCUT*loss_conf_nocut + LambdaCUT*loss_cut + LambdaVIS*loss_conf_vis
* 結論： Overfitting，loss 都往loss_coordinate學去，LambdaNOCUT*loss_coord的值太大

---------------------------------------------------------------------------------------------------------------------------------
[version 13]
* 改進criterionLV()， loss/數量 (平均)
*結論：Cvis幾乎都等於1，Cnocut幾乎都小於0.5，模型認為所有點都是cut但visible => 矛盾

---------------------------------------------------------------------------------------------------------------------------------
[version 14] 
* 改進criterionLV()，增加 loss occlded，都是使用MAELoss()
* loss = LambdaNOCUT*loss_coord + LambdaNOCUT*loss_conf_nocut + LambdaCUT*loss_cut + LambdaVIS*loss_conf_vis + 【LambdaOCC * loss_occ】
* 參數:
-LambdaCOOR = 20
-LambdaNOCUT = 1
-LambdaCUT = 11.6
-LambdaVIS = 0.25
-LambdaOCC = 0.75
* 結論: 幾乎所有點都是cut 且 invisible 

---------------------------------------------------------------------------------------------------------------------------------
[version 15] 
* 使用 BCELoss() 做 confidence_nocut和 confidence_vis
* loss = loss_coord + BCELoss(conf_nocut) + BCELoss(conf_vis)
* 結論： 結果有點根據，但預測visible/occlded結果沒有很好
* Evaluate: epoch80 Average NMSE = 0.017730970006699008

---------------------------------------------------------------------------------------------------------------------------------
[version 16] 
* 使用FOCALLoss() 做 confidence_nocut和 confidence_vis
* loss = loss_coord + FOCALLoss(conf_nocut) + FOCALLoss(conf_vis)
* 結論: confidence_nocut 和 confidence_vis 都等於 0.5 （r的值不小心給錯了，FOCALLoss(conf_nocut, r=1) / FOCALLoss(conf_vis, r=2)）
* 改: r_nocut = r_vis = 1 => 結果一樣

---------------------------------------------------------------------------------------------------------------------------------
[version 17] 
* 嘗試先訓練 visible or occluded
* 使用FOCALLoss() 做 confidence_vis
* loss = MAE(loss_coord) + FOCALLoss(conf_vis)
* 結論: landmark visible無根據 0.5值大多在左右
* 改1: network output conf_nocut部份拔掉 => 結果差不多
* 改2: 用BCELoss()做 confidence_vis => landmark visible 準確率提高許多 

---------------------------------------------------------------------------------------------------------------------------------
[version 18] 改version 8
1. 把heatmap預處理改成gaussian => sigma = (h+w)/30
2. 使用DenseNet pretrained model
3. gt_vis => vis=1, occ=0.5
4. loss = gt_vis*MSE(heatmap)
* 結論： heatmap result學壞，不知道是不是gt_heatmap的預處理有問題，所以再嘗試用11*11大小的方塊做heatmap預處理試試看

---------------------------------------------------------------------------------------------------------------------------------
[version 19] 改version 18
* train_bbox
* 嘗試用11*11大小的方塊做heatmap預處理試試看

---------------------------------------------------------------------------------------------------------------------------------
[version 20] 
* train_heatLV
* Network: HeatLVNet()
* Loss: BCE(vis) + w_vis*MSE(heat)
* vis => vis=1, occ=0
* w_vis => vis=1, occ=0.5

---------------------------------------------------------------------------------------------------------------------------------
[version 21] 
* train_heatLV
* Network: HeatLVNet()
* Loss: BCE(vis) + MSE(heat)
* vis => vis=1, occ=0
* heat: gaussian => (bbox_w+bbox_h)//70

---------------------------------------------------------------------------------------------------------------------------------
[version 22] 
* train_Unet
* Network: LVUNet
* Loss: BCE(vis) + MSE(heat)
* vis => vis=1, occ=0
* heat: gaussian => (bbox_w+bbox_h)//70
* 結論: landmark NMSE 結果比 HeatLVNet() 好，但visibility還是差

[version 22-2]
加dropout

[version 22-3]
BCE(vis)*10 + MSE(heat)

---------------------------------------------------------------------------------------------------------------------------------
[version 23] 
* train_Unet
* Network: LVUNet2 				#concat input downsample to feature(28*28) as information of visibility
* Loss: BCE(vis) + MSE(heat)
* vis => vis=1, occ=0
* heat: gaussian => (bbox_w+bbox_h)//70
* 沒有很好

---------------------------------------------------------------------------------------------------------------------------------
[try] 
* train_Unet
* Network: LVUNet3				# Use 2 UNet Architecture
* 結論: BAD

---------------------------------------------------------------------------------------------------------------------------------
[v24]
* train_Unet
* Network: LVUNet4				# predict visibility use fc layer and change channel to 128
* Loss: FOCAL(vis) + MSE(heat)
* 結論: 0.3269 / 0.5145

---------------------------------------------------------------------------------------------------------------------------------
[version 25] 
* train_Unet
* Network: LVUNet5				# 改output activation function: tanh()
* Loss: MSE(heat)
* heat: vis=>1 ,  occ=>-1
* 結論：用heatmap-based的方式預測visibility效果很好！

---------------------------------------------------------------------------------------------------------------------------------
[version 26] 
* train_Unet
* Network: CUNet()
* input: (256, 256, 3)
* output: Heatmap(64, 64, 3)
* Loss: MSELoss(heat)
* heat_gt: vis=>gaussian(1), vis=>gaussian(-1)
* 結論：NMSE better, vis_acc worse

---------------------------------------------------------------------------------------------------------------------------------
[version 27] 
* train_UNETandGHCU
* Network: LVUNet_GHCU()
* Loss: MSELoss(heat) + MSELoss(loc) + BCELoss(vis)

[version 27-1] 先tanh再GHCU
* train_UNETandGHCU
* Network: LVUNet_GHCU2()
* Loss: MSELoss(heat) + MSELoss(loc) + BCELoss(vis)

[version 27-2] no regression vis
* train_UNETandGHCU
* Network: LVUNet_GHCU3()
* Loss: MSELoss(heat) + MSELoss(loc)

[version 27-3] two stage
* train_UNETandGHCU
* Network: LVUNet5() + GHCU()
* Loss: MSELoss(loc) + BCELoss(vis)
* 結果：REGRESSION location and HEATMAP visibility is better







































