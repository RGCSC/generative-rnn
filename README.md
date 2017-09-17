# Generative Code
### Hack the North 2017 Team Project
RGCSC-ML (Rudy Ariaz, Yijia Chen, Aaron Choo, and Charles Zhang)

This project uses recurrent neural networks and long short-term memory techniques (implemented through keras) to attempt to generate Python code from text. In particular, users should be able to describe a short problem (for example, an algorithm contest problem, such as one from [Codeforces](codeforces.com)), and the model should be able to produce code that effectively solves that problem.

The model will be trained on part of OpenAI's Description2Code data (found [here](https://openai.com/requests-for-research/#description2code)), which consists of 5000 input/output-based contest problems (only 79 "easy" problems will be used for training), consisting of a problem description and up to 50 examples of solutions (in Python) for each problem. The model will be trained on FloydHub (GPU) for approximately two hours, with parameters subject to experimentation.

Update (2017-09-17.07:00): Due to an error when creating the original training model, the objective of the project has changed to simply generate random code snippets instead of generating code based on textual descriptions. The model was trained in batches of 32 in 10 epochs of only the first problem (training data), but has very low accuracy (usually around 0%).
