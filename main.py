from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy
from spacy.training.example import Example
from tqdm import tqdm

TRAIN_DATA = [
    ('Who is Nishanth?', {'entities': [(7, 15, 'PERSON')]}),
    ('Who is Kamal Khumar?', {'entities': [(7, 19, 'PERSON')]}),
    ('I like London and Berlin.', {'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]})
]

model = None
output_dir = Path("models/")
n_iter = 100

if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

# Set up the pipeline
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])  # Add entity labels

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # Only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            # Create Example objects
            example = Example.from_dict(nlp.make_doc(text), annotations)
            # Update the model with Example objects
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)


if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)