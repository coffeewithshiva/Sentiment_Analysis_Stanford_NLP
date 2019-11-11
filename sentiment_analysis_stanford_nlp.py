# -*- coding: utf-8 -*-
"""
Navigate to your stanford NLP path and run the below command in the command prompt.
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

"""

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

text = "This movie was actually neither that funny, nor super witty. The movie was meh. I liked watching that movie. If I had a choice, I would not watch that movie again."
text = "What amazing service Apple won't even talk to me about a question I have unless I pay them $19.95 for their stupid support!"
results = nlp.annotate(text,properties={
        'annotators':'sentiment, ner, pos',
        'outputFormat': 'json',
        'timeout': 50000,
        })

for s in results["sentences"]:
    print("{} : {}".format(" ".join(t["word"] for t in s["tokens"]),s["sentiment"]))
