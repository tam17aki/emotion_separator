# A neural network to separate emotional components from x-vectors

x-vectorから感情成分を分離するニューラルネット

感情成分の分離のために，感情分類に敵対的な学習を取り入れた．

## Recipes
1. Modify `config.yaml` according to your environment. root_dirの修正のみで十分．

2. Run `preprocess.py`. 前処理を実行する． 具体的には，声優統計コーパスのダウンロード（および解凍），x-vector抽出のための事前学習済モデルのダウンロード，および声優統計コーパスの音声群からx-vectorを抽出して保存する処理からなる．

3. Run `training.py`. モデルの訓練を実行する．
