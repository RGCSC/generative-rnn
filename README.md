# Generative Code
## Hack the North 2017 Project

This project uses machine learning concepts (implemented through keras) to attempt to generate Python code from text. In particular, users should be able to describe a short problem (for example, an algorithm contest problem), and the model should be able to produce code that effectively solves the problem.

The model will be trained on OpenAI's Description2Code data, which consists of 5000 input/output-based contest problems, consisting of a problem description and several examples of solutions (in Python) for each problem. The model will be trained on FloydHub (GPU) for at least two hours, with parameters subject to experimentation.
