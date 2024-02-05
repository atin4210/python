# Deep Learning: Mastering Neural Networks
## Capstone Project: Music Classifier
### Problem Statement

Classify songs of music in terms of its mood: Mellow (class 0) or Not Mellow (class 1). In the future other moods will be added.

### Info
1. For the CNN of the Music Classifier to work, you need to have a CSV file and image files.
    a.  Input CSV file (image_file_path,genre,mood): 
        DEFAULT: /content/drive/MyDrive/MIT_DL_Project_Data/melspec/music_melspec_data.csv

        CSV Fields:
        Image_file-path: file path to the image file
        Genre: [0 - 7] -- not used in the Capstone project
        Mood: [0, 1] -- 0 == Mellow music, 1 == Not Mellow music

    b.  Image files in some folder. The default is Mel Spectrograms files of size 1600x400 pixels as PNG (RGBA) files.
        DEFAULT: /content/drive/MyDrive/MIT_DL_Project_Data/melspec/

### Files & Folders
1.  src/capstone/music_classifier -- Project code and data
    a.  MusicClassifier.ipnyb -- is the Capstone Project code file
    b.  melspec/ -- contains the Mel Spectrogram (melspec) image files and music_melspec_data.csv
        i.   music_melspec_data.csv -- image,genre,mood
    c.  mfcc/ -- contains the Mel-Frequency Cepstrum Coefficients (mfcc) image files and the music_mfcc_data.csv
        i.   music_mfcc_data.csv -- image,genre,mood
    d.  tmp-dev/ -- folder for temporary development files
        i.   spectrogram_generator.py -- generator for Mel and MFCC Spectrograms. 
        ii.  music.csv -- song_filepath,genre,mood -- this is the input file for the spectrogram generator
        iii. temp_project.py -- initial project development file for my laptop.

### Contact
Please email [atin4210@gmail.com](mailto:atin4210@gmail.com) if you have any questions.
