# Deep Learning: Mastering Neural Networks
## Capstone Project: Music Classifier
Atin Kothari
February 03, 2024

### Proposal
This is a music classification project that uses a Convolution Neural Network (CNN) to analyze music and determine its mood. Its job is to classify music into the Mellow or Not Mellow moods. Eventually, it will become a more robust CNN that can classify in multiple categories, for example in Mood and Genre.

### Domain Background
Music/Audio Analysis is an ongoing research area and application of the Music Information Retrieval (MIR) field. 

### Problem Statement
I am proposing using of visual representation/s of audio/music to perform analysis on. ideally, we would use spectrograms, chromagrams, tempograms and waveforms for the analysis, but given the 3 week timeframe for the project proposal and completion, along with my lack of knowledge of MIR, I will use a single image of a Mel-spectrogram per song for it's Mood classification into Mellow or Not Mellow.

### Datasets and Inputs
The dataset consists of data from 410 music files from my personal collection. For each song, 130 seconds of music will be converted into PNG images (RGBA) of mel-spectrograms. I found that doing classification of publicly available music datasets was going to take a lot longer than I would have liked.

### Solution Statement
I will use a CNN with a high number of channels to capture that large set of details in each mel-spectrogram. 

### Evaluation Metrics
A confusion matrix will show the classification success rate, along with the Training curves for Loss and Accuracy. Accuracy of over 95% would be a great result for this project.

### Project Design
The project is divided into two main sections, the first is Data Preparation and the second is the CNN Model Training and Testing.

For Data Preparation, I will have to identify songs for classification by listening to them, perform MIR from them, create the CSV file and manually classify each song.

For the CNN Model Training and Testing, I will initially do it on my laptop until the approximate results show that the model is correct. Since my laptop is not powerful, tweaking of the hyper-parameters will have to be done in Google Colab.
