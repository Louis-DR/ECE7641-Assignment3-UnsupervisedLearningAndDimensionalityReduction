CS-7641
Unsupervised Learning and Dimensionality Reduction
Louis DURET-ROBERT



1. download the datasets and the git repository :
    - https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv
    - https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    - https://github.com/Louis-DR/SupervisedLearning
2. extract the archives and rename the folders to get the following tree :
   | wine-quality
   |   | wine_red.csv
   |   | wine_white.csv
   | wine-reviews
   |   | winemag-data_first150k.csv
   |   | winemag-data-130k-v2.csv
   | analysis.py
   | utils
   | main.py
   | scraper.py
3. install Python 3.7.4 and the following libraries :
    - numpy
    - scipy
    - matplotlib
    - pandas
    - sklearn
    - importlib
    - mca
    - keras
4. The best way to view and execute the code is to install Jupyter, Visual Studio Code and the Jupyter or Microsoft Python extension
4bis. To view the colored sections, install the extension Colored Regions by mihelcic and add this to the user settings JSON :
    "coloredRegions.namedColors": {
        "danger": "rgba(255, 0, 0, 0.5)",
        "warning": "rgba(255, 150, 0, 0.5)",
        "cyan": "rgba(26, 188, 156,0.1)",
        "green": "rgba(46, 204, 113,0.1)",
        "blue": "rgba(52, 152, 219,0.1)",
        "purple": "rgba(155, 89, 182,0.1)",
        "yellow": "rgba(241, 196, 15,0.1)",
        "orange": "rgba(230, 126, 34,0.1)",
        "red": "rgba(231, 76, 60,0.1)",
        "white": "rgba(236, 240, 241,0.1)",
        "black": "rgba(0, 0, 0,0.2)"
    }
5. Open main.py
6. Execute the first cell to imports the libraries
7. Execute the second or third cell to load either dataset. You can change the parameters of the function for a larger or smaller sample of the dataset
8. Read the comments and print() to execute the wanted cells in order :
    - The yellow and green cells correspond to K-means and Expectation maximization. You can change the range of number of clusters
    - The red, orange, purple and blue cells correspond to PCA/MCA, ICA, RP and the autoencoder/deep autoencoder. You can change the parameters of each algorithm : the number of components to keep, the number of dimensions in each layer, etc
    - The next yellow and green cells also correspond to K-means and EM but are used to add to the dataset the cluster information before running the neural network
    - The final black cell is the neural network
(be carefull, if you are on Windows, some of the cells emit a beeping sound to notify when they are finished)