# A neural network to separate emotional components from x-vectors

## 概要
x-vectorから感情成分を分離するニューラルネットの実装を提供する．

感情成分の分離のために，感情分類に敵対的な学習を取り入れた．

## 使用データ
声優統計コーパス https://voice-statistics.github.io/

## Recipes
1. Modify `config.yaml` according to your environment. root_dirの修正のみで十分．

2. Run `preprocess.py`. 前処理を実行する． 具体的には，声優統計コーパスのダウンロード（および解凍），x-vector抽出のための事前学習済モデルのダウンロード，および声優統計コーパスの音声群からx-vectorを抽出して保存する処理からなる．

3. Run `training.py`. モデルの訓練を実行する．

4. Run `inference.py`. 訓練済みのモデルにx-vectorを通すことで，感情成分が除去されたx-vectorを手に入れ，保存する．
5. Run `plot_umap_all.py`. UMAPによりx-vectorを可視化する．
