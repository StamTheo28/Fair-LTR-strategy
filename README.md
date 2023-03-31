# Fair Learning-to-Rank Strategy

## Name
Evaluating Fair Learning-to-Rank Strategies for Wikipedia Articles


## Description
This dissertation investigates the bias in the Wikipedia search engine, where editors receive
articles with which are primarily based on certain dominant characteristics, that can result in
under-exposed protected groups receiving an unfair exposure. The goal of this study is to develop
and implement seven strategies that will use learning to rank to decrease this unfairness without
reducing the relevance of articles assigned to editors from a vertical ranked list. The study employs
feature selection and feature boosting techniques during the post-processing phase of machine
learning to promote fairness. Although the results did not yield statistical significance, the study
identified a single best algorithm for boosting fairness and detecting which features have a higher
importance in the learning to rank model using the Trec-Fair 2022 dataset.


## Installation
In order to use our implementation you much use a linux Operating System as PyTerrier is not compatible with a Windows based O.S. The you Firstly have to clone our repository using the following command 
- 'git clone https://stgit.dcs.gla.ac.uk/2380138t/fair-learning-to-rank-strategy.git'

Once this is done, you have to install PyTerrier on your machine using:
- 'pip install python-terrier'

Then if you go to the directory fair-learning-to-rank-strategy/Fair-Ranking-LTR and run the main.py file then the experiment will begin.
The experinmental process goes as follows:
- Initialise a PyTerrier Instance
- Index dataset
- Compute Distributional Fairness Scores
- Train the 7 strategies
- Evaluate the 7 strategies based on relevance and fairness




