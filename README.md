# LMIR

Pure Python implementations of the language models for information retrieval surveyed [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.8019&amp;rep=rep1&amp;type=pdf).

```python3
import lmir

doc_1 = "This is document one.".split()
doc_2 = "This is document two. It contains different words.".split()
docs = [doc_1, doc_2]

models = lmir.LMIR(docs)

print(models.jelinek_mercer("This query has words that are found in the corpus.".split()))
print(models.jelinek_mercer("No matches.".split()))
```
