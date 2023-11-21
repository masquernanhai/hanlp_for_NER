import hanlp
import os
import hanlp
from hanlp.components.ner.transformer_ner import TransformerNamedEntityRecognizer

root_path = os.path.join(os.path.dirname(__file__),'..')
training_corpus = os.path.join(root_path, 'data/finetune/ner/train.tsv')
development_corpus = os.path.join(root_path, 'data/finetune/ner/dev.tsv')  # Use a different one in reality

model_name = 'msra_ner_electra_small_20220215_205503'
save_dir = os.path.join(root_path, 'model/{}'.format(model_name))

if not os.path.exists(training_corpus):
    os.makedirs(os.path.dirname(training_corpus), exist_ok=True)
    with open(training_corpus, 'w') as out:
        out.write(
'''训练\tB-NLP
语料\tE-NLP
为\tO
IOBES\tO
格式\tO
'''
        )

ner = TransformerNamedEntityRecognizer()
ner.fit(
    trn_data=training_corpus,
    dev_data=development_corpus,
    save_dir=save_dir,
    epochs=50,  # Since the corpus is small, overfit it
    finetune=hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH,
    # You MUST set the same parameters with the fine-tuning model:
    average_subwords=True,
    transformer='hfl/chinese-electra-180g-small-discriminator',
)

