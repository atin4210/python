# Deep Learning: Mastering Neural Networks
## Capstone Project: Music Classifier
Atin Kothari
February 03, 2024

### Proposal
Create a Music Classifier using a Convolution Neural Network (CNN) to analyze music and determine its mood. Its job is to classify music into the 'Mellow' or 'Not Mellow' moods. Eventually, it will become a more robust CNN that can classify in multiple categories, for example in Mood and Genre.

### Domain Background
Music/Audio Analysis is an ongoing research area and application of the Music Information Retrieval (MIR) field. 

### Problem Statement
I am proposing using of visual representation/s of audio/music to perform analysis on. Ideally, I would use spectrograms, chromagrams, tempograms and waveforms for the analysis, because they provide very different views of the same data, but at present I don't know how to do it. Given the 3 week timeframe for the project proposal and completion, along with my lack of knowledge of MIR, I am going to use a single image of a Mel-spectrogram per song for it's Mood classification into Mellow or Not Mellow.

### Datasets and Inputs
The dataset consists of data from 410 music files from my personal collection. For each song, 130 seconds of music will be converted into PNG images (RGBA) of mel-spectrograms. I found that doing classification of publicly available music datasets was going to take a lot longer than I would have liked.

### Solution Statement
The solution is a CNN with multiple Convolutional and Pooling layers, with a high number of channels to capture that large set of details in each mel-spectrogram. 

### Evaluation Metrics
For evalutation, I will use the Confusion Matrix to show the classification success rate, along with the Training curves for Loss and Accuracy. Accuracy of over 95% would be a great result for this project.

### Project Design
The project is divided into two main sections, the first is Data Preparation and the second is the CNN Model Training and Testing.

For Data Preparation, I have to do the following:
1. identify songs for classification by listening to them, 
2. write a shell script to generate a CSV file containing the song filename, genre and mood, 
3. write a Python program using Librosa to do MIR, have it generate the Mel-spectrogram image for each song and generate a CSV file,
4. manually classify each song in the CSV file.

For the CNN Model Training and Testing, I will have to do the following:
1. create and test the CNN prototype on my laptop until the results show that the model is correct,
2. push the files to Google Colab for the tweaking of the hyper-parameters and final testing

The final CNN model is as follows:

    MusicClassifier(
    (conv0): Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu):  ReLU()
    (pool):  MaxPool2d(kernel_size=2, stride=2)
    (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu):  ReLU()
    (pool):  MaxPool2d(kernel_size=2, stride=2)
    (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu):  ReLU()
    (pool):  MaxPool2d(kernel_size=2, stride=2)
    (conv3): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu):  ReLU()
    (pool):  MaxPool2d(kernel_size=2, stride=2)
    (fc1):   Linear(in_features=640000, out_features=512, bias=True)
    (relu):  ReLU()
    (dropout): Dropout(p=0.5, inplace=False)
    (fc2):   Linear(in_features=512, out_features=2, bias=True)
    )

Since the images are 1600x400 pixels in size and contain a lot of detail, the CNN contains 256 out-channels for the last of 4 Convolution layers. There are 4 MaxPool2D layers to scale down the images to a manageable size for linearizing. Even though the dataset is very small (410 images), but the images are large (1600x400 pixels), I had to use a batch size of 8, because any larger (like 16 or more) was causing the GPUs to run out of memory. As an added benefit, the small batch size introduced a little bit of noise in the gradient estimation and thereby providing implicit regularization.
