## AI playbook

The playbook is a collection of AI projects ranging from Computer Vision to Natural Language Processing:

##### Reinforcement Learning:

* [Reinforcement Learning Racing](modules/reinforcement-learning-racing)
(includes random racing circuit generator through b-splines using pygame with agent acting on it using either DDQN, A3C or PPO)

##### Natural Language Processing:

* [Speech Recogniser](modules/speech-recogniser)
(implements a CNN + RNN architecture with CTCLoss for end-to-end speech recognition)

* [POS Tagging](modules/pos-tagging)
(probabilistic graphical model(bayesian) such as HMM and MEMM, and Viterbi algorithm for decoding)

* [NER Tagging](modules/ner-tagging)
(NN CRF-layer implementation, work in progress...)

* [Word Embeddings](modules/word-embeddings)
(implements word embeddings generation such as Word2Vec and GloVe)


##### Computer Vision:
* [Flower Classifier](modules/flower-classifier)
(includes transfer learning for flower classification with 99% accuracy on 
[dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html))

* [Dog Classifier](modules/dog-classifier)
(includes both custom CNN & transfer learning and for dog classification)

* [Semantic Segmentation](modules/semantic-segmentation)
(uses MobileNetV2 with custom decoder to generate a semantic segmentation on COCO dataset)

* [Style Transfer](modules/style-transfer)

##### Misc:

* [Auction Sale Price Prediction](modules/auction-sale-price)
(uses Random Forest and extra pre-processing such as feature importance and engineering)

* [Audio Signal Notes](modules/audio-signal)

* [Bike Sharing Prediction](modules/bike-sharing)
(includes DNN built in NumPy for bike sharing prediction)

* [Collaborative Filtering](modules/collaborative-filtering)
(uses collaborative filtering to predict similar music recommendation based on similar interest of his peers)

* [Random Forest](modules/random-forest)
(random forest implementation in Pandas and NumPy)

* [Disease Linkage](modules/collaborative-filtering)
(implements ridge regression to detect multicollinearity among patient's characteristics leading to prostate cancer)

* [Track Recogniser](modules/track-recogniser)
(implements generation of track's fingerprints through storing the most potent frequency bands with O(1) retrieval through hashing)


### Disclaimer
The given structure is chosen as it offers better flexibility to make it more modular. Any feedback is greatly appreciated.