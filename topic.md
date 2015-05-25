# 4章 トピックモデル

- 前章では1つのクラスに各データを割り当てた
- この章では、トピック(topic) と呼ばれる小さなグループに割り当てる
- データの中心的な話題とそれ程重要でない話題を見分ける方法についても学ぶ

## 4.1 潜在的ディリクレ配分法 (LDA)

**LDA**
1. Latent Dirichlet Allocatioin 潜在的ディリクレ配分法 (以下*LDA*をこちらの意味で用いる)
2. Linear Discriminant Analysis 線形判別分析 (scikit-learn にある sklearn.lda はこっち)

- LDA は最も単純なトピックモデルであり、多くのトピックモデル手法の基礎
- LDA の背景にある数式については [Wikipedia のページ](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) を参照

トピックモデルのおおまかなイメージ
- 「文章製造機」 がモデルの背後にある
- この文章製造機の中には、固定されたトピックがある

- 例) 3つのトピック
    - 機械学習
    - Python
    - 料理

- 各トピックは関連する単語のリストを持つ
- もしこの本が「機械学習」「Python」について 50% ずつ書かれているとすると
  その2つのトピックの持つ単語リストから、この本で使われている単語が半分ずつ選ばれたことになる   

実際には、私たちはトピックが何であるかを知らない
解くべき問題は、「文章製造機」のリバース・エンジニアリング
-> どのようなトピックが存在し、各文書がどのトピックに割り当てられているかを解明すること

### 4.1.1 トピックモデルの作成

前述の通り scikit-learn には LDA のモジュールはないので、gensim (by Radim Rehurek) パッケージを使う

- gensim のインストール
`pip install gensim` または `easy_install gensim`

まず始めに、Associated Press (AP) データセット（トピックモデル研究の初期から使われている、ニュースレポートのデータセット）を用いて、トピックモデルを作成する

    from gensim import corpora, models, similarities
    corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

以下のコードでトピックモデルの作成が完了

    model = models.ldamodel.LdaModel(
    corpus,
    num_topics=100,
    id2word=corpus.id2word)

- トピック数は 100 に指定

ある文書がどのようなトピックを持つかを見てみる

    topics = [model[c] for c in corpus]
    print topics[0]

    [(3, 0.023607255776894751),
      (13, 0.11679936618551275),
      (19, 0.075935855202707139),
      (92, 0.10781541687001292)]


- `model[doc]` は doc が持つトピックを (topic_index, topic_weight) のリストの形で返す
- 上ではコーパス中の最初の文書が持つトピックのリストを出力させている
- ある文書について返されるトピックの数は 100 個中数個
- つまり、各文書は一部のトピックだけから構成されている -> トピックモデルは疎なモデル

![トピック数](https://github.com/poiuiop/machine_learning_python/blob/master/img/01.png)

- 約150個の文書が 5つのトピックを持っている
- 10 ~ 12個のトピックを持っている文書も多くある
- 20 個以上のトピックを持つ文書はほとんどない

各文書が持つトピック数を増やすには、モデルの作成時の関数で、alpha パラメータの値を指定する
- alpha が大きくなると各文書が持つトピック数が増える
- alpha > 0, 通常は小さな値(< 1), gensim のデフォルトは 1.0 / len(corpus)

    model = models.ldamodel.LdaModel(
    corpus,
    num_topics=100,
    id2word=corpus.id2word,
    alpha=1)


![alpha別](https://github.com/poiuiop/machine_learning_python/blob/master/img/02.png)

- alpha = 1.0 にすると、多くの文書が 20 ~ 25個のトピックを持つようになる

#### トピックの持つ意味

単語についての多項分布 -> 各単語に確率を与える。確率の高い単語はそのトピックと関連性が高い
トピックを要約するためには、高い確率を持つ単語のリストを提示するのが一般的

![初めの10個のトピック](https://github.com/poiuiop/machine_learning_python/blob/master/img/03.png)
![初めの10個のトピック](https://github.com/poiuiop/machine_learning_python/blob/master/img/04.png)

- 当然ながら、トピックが持つ単語同士には関連性がある
- 単語には重要度が与えられているので、ワードクラウドによっていい感じに表示できる

![ワードクラウド](https://github.com/poiuiop/machine_learning_python/blob/master/img/05.png)

- ストップワードの除去やステミングも重要

## 4.2 トピック空間で類似度の比較を行う

トピックはそれだけでも実用的
- ワードクラウドのような形で文書の要約が作成できる
- 大量に文書がある場合のナビゲーションになる

ここまでやってきたこと
-> 各文書が各トピックからどれくらいの割合で生成されているかということの予測

この応用として、トピック空間での文書の比較を行う
- 2つの文書が同じトピックについて論じているなら、それらの文書は似ている
- 共通する単語があまりなくても、同じトピックを扱っている場合がある = 異なる表現で同じようなことを述べている

前章で扱った問題にトピックモデルを適用する

- 単語の頻度ベクトル -> トピックベクトル
- 文書をトピック空間に射影し、各トピックの重みベクトルで表現
- トピック数 (今は100) は単語数より少ないので、次元削減になる

先程得た、topics (corpus 中の各文書が持つトピック) の値を NumPy の配列に格納する

    dense = np.zeros( (len(topics), 100), float)
    for ti,t in enumerate(topics):
        for tj,v in t:
            dense[ti,tj] = v

ti 番目と tj 番目の距離は、`sum((dense[ti] - dense[tj])**2)`で計算できる
pdist 関数を用いると、文書の全ての組み合わせでこの計算ができる

    from scipy.spatial import distance
    pairwise = distance.squareform(distance.pdist(dense))

ここで、距離行列 (pairwise) の対角要素に大きな値を設定する必要がある

    largest = pairwise.max()
    for ti in range(len(topics)):
        pairwise[ti,ti] = largest + 1

- 距離行列における最も大きな値 (largest) より大きな値にする（1 を足している）
- 対角要素には引数自身との距離が入っているので、そこに大きな値を入れておかないと以下の関数は常に引数と同じ ID の文書を返してしまう

    def closest_to(doc_id):
        return paiirwise[doc_id].argmin()

この関数を用いると、例えばデータセットの二つ目の文書に最も類似する文書は`closest_to(1)`で取れる

![二つ目の文書](https://github.com/poiuiop/machine_learning_python/blob/master/img/06.png)
![最も類似する文書](https://github.com/poiuiop/machine_learning_python/blob/master/img/07.png)

- 両方とも同じ人が書いた文書で薬について書かれている

4.2.1 Wikipedia 全体のモデル化

Wikipedia の英語記事全体を対象にトピックモデルを作成してみる
[Wikipedia のダンプファイル](http://dumps.wikimedia.org) をダウンロード

gensim を用いてインデックス化

    python -m gensim.scripts.make_wiki enwiki-latest-pages-articles.xml.bz2 wiki_en_output

必要なパッケージをインストール

    import logging, gensim
    logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

前処理をしたデータの読み込み

    id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_output_wordids.txt')
    mm = gensim.corpora.MmCorpus('wiki_en_output_tfidf.mm')

モデルの作成

    model = gensim.models.ldamodel.LdaModel(
    corpus=mm,
    id2word=id2word,
    num_topics=100,
    update_every=1,
    chunksize=10000,
    passes=1)

結果をファイルに保存

    model.save('wiki_lda.pkl')

- ダウンロード、インデックス化、モデル作成には非常に時間がかかる（それぞれ数時間）
- 保存しておけばここから続きができる

    `model = gensim.models.ldamodel.LdaModel.load('wiki_lda.pkl')`

    ```topics = []
    for doc in mm:
        topics.append(model[doc])```
    
    ```import numpy as np
    lens = np.array([len(t) for t in topics])```

- lens -> 各文書のトピック数のリスト

    print np.mean(lens)

    6.55842326445

- トピック数の平均値 -> およそ 6.5 （疎なモデル）

    print np.mean(lens <= 10)

    0.932382190219

- トピック数 10個以下の割合 -> およそ 93%

データ全体での各トピックの使用回数をカウント

    counts = np.zeros(100)
    for doc_top in topics:
        for ti,_ in doc_top:
            counts[ti] += 1
    words = model.show_topic(counts.argmax(), 64)

- book, movie, fiction, story などが多い

最も書かれることが少ないトピック

    words = model.show_topic(counts.argmin(), 64)

## 4.3 トピックの数を選択する

ここまではトピック数は 100 に固定していた
- 変更してもよいが、レコメンドシステム等の中間的な要素としてトピックモデルを用いる場合、トピック数の多寡はあまり重要ではない
- ある一定数のトピックに話題が集中しているということか？
- 100 程度のトピック数があれば十分
- alpha の値も同様

しかし、データセットに応じてパラメータの値を自動で決める手法は存在する
- その一つとして、階層ディリクレ過程 (hierarchical Dirichlet process : HDP) がある
- トピック数を固定し、それに合うようにデータのリバース・エンジニアリングを行うのではなく、データに従ってトピックを生成する
- イメージとしては、物書きが新しい文書を書く時に、既存の話題を使うか新しい話題を作り出すかのオプションを持っているようなもの

- この方法では、文書が多いほど、より多くのトピックを得ることができる
- 文書が増えるほど、話題を細かく認識できるようになるため
- 「スポーツ」-> 「ホッケー」「サッカー」など

HDP は gensim に用意されている

    hdp = gensim.models.hdpmodel.Hdpmodel(mm, id2word)

- モデル作成の部分を変えるだけ

## 4.4 まとめ

- トピックモデルを用いると、各文書を一つ以上のグループに所属させられる（単純なクラスタリングとの違い）
- トピックモデルの基本的な手法である LDA は gensim で簡単に使える
- トピックモデルは比較的新しい分野
- 最近ではコンピュータビジョンにおいて特に重要になっている
