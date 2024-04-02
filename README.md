An SVM that tries to predict IMDB scores using the reviews. When run, it runs 4 times with 4 different datasets:

1. Only reviews with 1s and 10s
2. Reviews with 1s, 5s, and 10s,
3. Reviews with 7s and 8s
4. Reviews with all scores from 1-10

The SVM scores poorly with all of the datasets except the first one, but feel free to run it anyways

The code is created in python. Use whatever environment is comfortable for you.

Required Libraries:
time
numpy
pandas
nltk
sklearn
seaborn
matplotlib

time, seaborn, and matplotlib aren't strictly necessary so you can remove them and any code that uses them if you'd like.
pandas should be updated to the lastest version 2.0.0 as I'm not 100% sure how pandas' concat function behaves in previous versions.

The code should take around 2-3 minutes to run, depending on your computer's power.