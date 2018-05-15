# graduation-thesis

### 概要
interimは中間発表用。
srcにはcomparison,data,main.resultの4つのディレクトリが含まれています。
testは提案手法を構成するための実験的なコードを含む。

* comparison:比較手法のソースコード
* data:データセットに用いたYahooニュースとLivedoorニュースの記事タイトルおよびカテゴリー
* main:提案手法と結果表示用、デモ用の3つのソースコード
* result:実験結果を記録したjsonファイル、提案手法の学習済みモデルを保存したファイル

### comparison
* BOW_News.ipynb:BOW表現を用いた分類
* BOW_tfidf_lda_News.ipynb:BOW+TFIDF+LDAを用いた分類
* C_LSTM_News.ipynb:C-LSTM
* CNN_News.ipynb:普通のCNN
* CNNSC_News.ipynb:CNN for sentence classification
* FFN_attention.ipynb:Feed Forward Networks with Attention
* NaiveBayes_News.ipynb:ナイーブベイズによる分類
* RCNN_News.ipynb:Recurrent Convolutional NN
* Word2Vec_Mean_News.ipynb:Bag-of-Meansを用いた分類

### data
* livedoor_data.txt:livedoorニュース記事タイトルとカテゴリー
* yahoo_data.txt:yahooニュース記事タイトルとカテゴリー

### main
* _Main_News.ipynb:提案手法による分類
* _PLOT_News.ipynb:結果表示用
* news_demo.py:デモ用のプログラム

### result
* livedoor.json:提案手法を用いてLivedoorニュース記事を分類した際の結果
* yahoo.json:提案手法を用いてYahooニュース記事を分類した際の結果
* lstm_attention.h5:提案手法における学習後のモデルの状態を保存したファイル
