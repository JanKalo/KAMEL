# KAMEL 🐫: Knowledge Analysis with Multitoken Entities in Language Models


## Abstract:
Large language models (LMs) have been shown to capture large amounts of relational
knowledge from the pre-training corpus. These models can be probed for this factual knowl-
edge by using cloze-style prompts as demonstrated on the LAMA benchmark. However,
recent studies have uncovered that results only perform well, because the models are good
at performing educated guesses or recalling facts from the training data. We present a novel
Wikidata-based benchmark dataset, KAMEL , for probing relational knowledge in LMs.
In contrast to previous datasets, it covers a broader range of knowledge, probes for single-,
and multi-token entities, and contains facts with literal values. Furthermore, the evaluation
procedure is more accurate, since the dataset contains alternative entity labels and deals
with higher-cardinality relations. Instead of performing the evaluation on masked language
models, we present results for a variety of recent causal LMs in a few-shot setting. We show
that indeed novel models perform very well on LAMA, achieving a promising F1-score of
52.90%, while only achieving 17.62% on KAMEL. Our analysis shows that even large lan-
guage models are far from being able to memorize all varieties of relational knowledge that
is usually stored knowledge graphs.

## Reference:

KAMEL is described in the following paper. The paper is available [here](https://www.akbc.ws/2022/assets/pdfs/15_kamel_knowledge_analysis_with_.pdf).
We also have a [poster](https://github.com/JanKalo/KAMEL/blob/master/poster_kalo.pdf).

```bibtex
@inproceedings{kalo2022kamel,
  title={KAMEL: Knowledge Analysis with Multitoken Entities in Language Models},
  author={Kalo, Jan-Christoph and Fichtel, Leandra},
  booktitle={Automated Knowledge Base Construction},
  year={2022}
}

}
```

### The dataset on the HuggingFace data hub.

*to be done*


## Reproducing our Results

*to be done*
