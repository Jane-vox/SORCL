# SORCL: Social-Reachability-driven Contrastive Learning for Friend Recommendation

## Datasets
We use Twitch, Deezer and Last.FM for evaluation. 

Datasets | #nodes  | #edges | #density
---- | ----- | ------ | ------
Twitch | 9,498  | 123,138 | 2.72e-3
Deezer |28,281 | 92,752 | 2.31e-4
Last.FM | 7,624 | 27,806 | 9.56e-4

Each dataset is composed of 5 files:
* `info.yaml`: training set information
* `indices.pkl`,`indptr.pkl`: training set
* `val.pkl`: validating set
* `test.pkl`: testing set


## Running the Model
To train and evaluate the model, please run

`python -m SORCL.main.run_model`
