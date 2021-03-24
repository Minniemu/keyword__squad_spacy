# Train with SQuAD & spaCy

* 使用之前的BERT-ner model做考眼生成

## Train
**此模型以 transformers 2.6.0 撰寫，請注意版本是否符合**
```
pip install transformers==2.6.0
```
* result2.csv 為 SQUAD dataset 的文章，並將其中之答案標註為 'A'，並另外使用spaCy將可以標註之名詞也在 result2.csv中標為 'A'。**此資料為訓練資料**
* bert_ner.py is for training


## Usage
* 因為 bert_ner.py 中有儲存訓練之model，因此我們在這邊可以直接使用 test.py 來使用訓練好的model
* test.py 是將 article.json 的文章拿出來，並使用剛訓練好的model，將 context、answer包成一個新的json 檔( qg_data.json , 稍後可做qg)



## ERROR Message
* Token indices sequence length is longer than the specified maximum sequence length for this model (537 > 512).
> 表示句子長度超過max_length
>![](https://i.imgur.com/GzFn86Q.png)

> Solution : 將超過 512 的句子強制截短
```
if len(test_sentence) > 512:
    test_sentence = test_sentence[:512]
```
