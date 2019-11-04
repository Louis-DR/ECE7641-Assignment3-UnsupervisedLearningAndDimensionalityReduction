#%%
print("----------Imports----------")
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.neural_network import MLPClassifier
from mca import MCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import calinski_harabasz_score

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from scipy.stats import kurtosis, kurtosistest
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-ticks')
plt.style.use('seaborn-whitegrid')

import time
from utils import chrono, tickinit, tick
from scraper import import_wine_review, import_wine_quality
from analysis import plot_confusion_matrix, plot_param_crossval_acc, analyse

import importlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import winsound
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
print("----------Done importing----------")



#%% #region[white]
print("----------Loading dataset----------")
XXX, yyy = import_wine_review(nbr_class=5, subset_size=10000, scalerr="centernorm")
X_train, X_test, y_train, y_test = train_test_split(XXX, yyy, test_size=0.1, random_state=0)
fold = 10
nbr_runs = 5
sample_dim = XXX.shape[1]
X_clust = XXX
XXX_cat = XXX.drop(['price','description_len'], axis=1)
print("----------Done loading dataset----------")
#endregion



#%% #region[white]
print("----------Loading dataset----------")
XXX, yyy = import_wine_quality(subset_size=0, scalerr="centernorm")
X_train, X_test, y_train, y_test = train_test_split(XXX, yyy, test_size=0.1, random_state=0)
fold = 4
nbr_runs = 5
sample_dim = XXX.shape[1]
X_clust = XXX
print("----------Done loading dataset----------")
#endregion



#%% region[yellow] K-MEANS
# ks = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75]
ks = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75]

min_sil = []
min_ari = []
min_ami = []
min_com = []
min_hom = []
min_chi = []

avg_sil = []
avg_ari = []
avg_ami = []
avg_com = []
avg_hom = []
avg_chi = []

max_sil = []
max_ari = []
max_ami = []
max_com = []
max_hom = []
max_chi = []

for k in ks:
    print("╔══════════════════════════╗")
    print("║ NUMBER OF CLUSTERS = {:<3} ║".format(k))
    print("╚══════════════════════════╝")
    
    sil = []
    ari = []
    ami = []
    com = []
    hom = []
    chi = []

    ticksize, tickmarks = tickinit(nbr_runs)
    chrono()
    for run_nbr in range(nbr_runs):
        km = KMeans(n_clusters=k, random_state=run_nbr).fit(X_clust)
        labs = km.labels_
        sil.append(silhouette_score(X_clust, labs))
        ari.append(adjusted_rand_score(yyy, labs))
        ami.append(adjusted_mutual_info_score(yyy,labs))
        com.append(completeness_score(yyy,labs))
        hom.append(homogeneity_score(yyy,labs))
        chi.append(calinski_harabasz_score(X_clust, labs))
        tick(ticksize, tickmarks, run_nbr)
    print("│  {0:.2f}s\n".format(chrono()))
    
    min_sil.append(np.amin(sil))
    avg_sil.append(sum(sil)/len(sil))
    max_sil.append(np.amax(sil))
    
    min_ari.append(np.amin(ari))
    avg_ari.append(sum(ari)/len(ari))
    max_ari.append(np.amax(ari))
    
    min_ami.append(np.amin(ami))
    avg_ami.append(sum(ami)/len(ami))
    max_ami.append(np.amax(ami))
    
    min_com.append(np.amin(com))
    avg_com.append(sum(com)/len(com))
    max_com.append(np.amax(com))
    
    min_hom.append(np.amin(hom))
    avg_hom.append(sum(hom)/len(hom))
    max_hom.append(np.amax(hom))
    
    min_chi.append(np.amin(chi))
    avg_chi.append(sum(chi)/len(chi))
    max_chi.append(np.amax(chi))

fig, ax = plt.subplots()
ax.plot(ks,avg_sil, color='RosyBrown', label='Silhouette', linewidth=3.0)
ax.fill_between(ks, min_sil, max_sil, color='RosyBrown', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Silhouette score')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ks,avg_ari, color='SteelBlue', label='Adjusted Rand Index', linewidth=3.0)
ax.plot(ks,avg_ami, color='DarkSeaGreen', label='Adjusted Mutual Information', linewidth=3.0)
ax.fill_between(ks, min_ari, max_ari, color='SteelBlue', alpha=0.15)
ax.fill_between(ks, min_ami, max_ami, color='DarkSeaGreen', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('ARI and AMI')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ks,avg_com, color='Goldenrod', label='Completeness', linewidth=3.0)
ax.plot(ks,avg_hom, color='MediumPurple', label='Homogeneity', linewidth=3.0)
ax.fill_between(ks, min_com, max_com, color='Goldenrod', alpha=0.15)
ax.fill_between(ks, min_hom, max_hom, color='MediumPurple', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Homogeneity and Completeness')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ks,avg_chi, color='LightSeaGreen', label='Calinski-Harabasz Index', linewidth=3.0)
ax.fill_between(ks, min_chi, max_chi, color='LightSeaGreen', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Calinski-Harabasz Index')
plt.legend()
plt.show()

winsound.Beep(frequency, duration)
#endregion



# %% region[green] EXPECTATION MAXIMIZATION

# ks = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,65,70,75]
ks = [2,3,4,5,6,7,8,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75]

min_sil = []
min_ari = []
min_ami = []
min_com = []
min_hom = []
min_chi = []

avg_sil = []
avg_ari = []
avg_ami = []
avg_com = []
avg_hom = []
avg_chi = []

max_sil = []
max_ari = []
max_ami = []
max_com = []
max_hom = []
max_chi = []

for k in ks:
    print("╔══════════════════════════╗")
    print("║ NUMBER OF CLUSTERS = {:<3} ║".format(k))
    print("╚══════════════════════════╝")
    
    sil = []
    ari = []
    ami = []
    com = []
    hom = []
    chi = []

    ticksize, tickmarks = tickinit(nbr_runs)
    chrono()
    for run_nbr in range(nbr_runs):
        labs = GaussianMixture(n_components=k, random_state=run_nbr).fit_predict(X_clust)
        sil.append(silhouette_score(X_clust, labs))
        ari.append(adjusted_rand_score(yyy, labs))
        ami.append(adjusted_mutual_info_score(yyy,labs))
        com.append(completeness_score(yyy,labs))
        hom.append(homogeneity_score(yyy,labs))
        chi.append(calinski_harabasz_score(X_clust, labs))
        tick(ticksize, tickmarks, run_nbr)
    print("│  {0:.2f}s\n".format(chrono()))
    
    min_sil.append(np.amin(sil))
    avg_sil.append(sum(sil)/len(sil))
    max_sil.append(np.amax(sil))
    
    min_ari.append(np.amin(ari))
    avg_ari.append(sum(ari)/len(ari))
    max_ari.append(np.amax(ari))
    
    min_ami.append(np.amin(ami))
    avg_ami.append(sum(ami)/len(ami))
    max_ami.append(np.amax(ami))
    
    min_com.append(np.amin(com))
    avg_com.append(sum(com)/len(com))
    max_com.append(np.amax(com))
    
    min_hom.append(np.amin(hom))
    avg_hom.append(sum(hom)/len(hom))
    max_hom.append(np.amax(hom))
    
    min_chi.append(np.amin(chi))
    avg_chi.append(sum(chi)/len(chi))
    max_chi.append(np.amax(chi))


fig, ax = plt.subplots()
ax.plot(ks,avg_sil, color='RosyBrown', label='Silhouette', linewidth=3.0)
ax.fill_between(ks, min_sil, max_sil, color='RosyBrown', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Silhouette score')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ks,avg_ari, color='SteelBlue', label='Adjusted Rand Index', linewidth=3.0)
ax.plot(ks,avg_ami, color='DarkSeaGreen', label='Adjusted Mutual Information', linewidth=3.0)
ax.fill_between(ks, min_ari, max_ari, color='SteelBlue', alpha=0.15)
ax.fill_between(ks, min_ami, max_ami, color='DarkSeaGreen', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('ARI and AMI')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ks,avg_com, color='Goldenrod', label='Completeness', linewidth=3.0)
ax.plot(ks,avg_hom, color='MediumPurple', label='Homogeneity', linewidth=3.0)
ax.fill_between(ks, min_com, max_com, color='Goldenrod', alpha=0.15)
ax.fill_between(ks, min_hom, max_hom, color='MediumPurple', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Homogeneity and Completeness')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(ks,avg_chi, color='LightSeaGreen', label='Calinski-Harabasz Index', linewidth=3.0)
ax.fill_between(ks, min_chi, max_chi, color='LightSeaGreen', alpha=0.15)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Calinski-Harabasz Index')
plt.legend()
plt.show()

winsound.Beep(frequency, duration)
#endregion



# %% region[red] PRINICPAL COMPONENT ANALYSIS
number_components = 11
pca = PCA(n_components=number_components)
X_pca = pca.fit_transform(XXX)

print(pca.explained_variance_ratio_) 
print(pca.singular_values_)
plt.plot(range(1,number_components+1), pca.explained_variance_ratio_, color="PaleVioletRed", label="Explained Variance Ratio", linewidth=3.0)
# plt.axis([0.8, number_components+.2, -0.01, 0.33])
plt.xlabel("Number of components")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.show()
plt.plot(range(1,number_components+1), pca.singular_values_, color="BurlyWood", label="Singular Values", linewidth=3.0)
# plt.axis([0.8, number_components+.2, -4, 140])
plt.xlabel("Number of components")
plt.ylabel("Singular values")
plt.legend()
plt.show()

colors = ['#D3D3D3', '#B0BFCB', '#8DABC4', '#6996BC', '#4682B4']

plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0,1,2,3,4], [1,2,3,4,5]):
    plt.scatter(X_pca[yyy == i, 0], X_pca[yyy == i, 1], color=color, lw=0.1, label=target_name)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

X_clust = X_pca
#endregion



# %% region[red] M COMPONENT ANALYSIS
mca = MCA(XXX_cat)
plt.plot(mca.expl_var(N=20), color="PaleVioletRed", label="Explained Variance Ratio", linewidth=3.0)
plt.xlabel("Number of components")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.show()

# X_mca = mca.fs_r(N=129)
X_mca = mca.fs_r(N=20)
X_clust = np.concatenate((np.array(X_pca),np.array(X_mca)),axis=1)
#endregion



# %% region[orange] INDEPENDANT COMPONENT ANALYSIS
ica = FastICA(n_components=645)
X_ica = ica.fit_transform(XXX)
# plt.plot(kurtosistest(XXX, axis=0).statistic)
# plt.plot(kurtosistest(X_ica, axis=0).statistic)

colors = ['#D3D3D3', '#B0BFCB', '#8DABC4', '#6996BC', '#4682B4']

plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0,1,2,3,4], [1,2,3,4,5]):
    plt.scatter(X_ica[yyy == i, 0], X_ica[yyy == i, 1], color=color, lw=0.1, label=target_name)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.xlabel("ICA1")
plt.ylabel("ICA2")
plt.show()

X_clust = X_ica
#endregion



# %% region[purple] RANDOM PROJECTIONS
rp = GaussianRandomProjection(n_components=100)
X_rp = rp.fit_transform(XXX)

colors = ['#D3D3D3', '#B0BFCB', '#8DABC4', '#6996BC', '#4682B4']

plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0,1,2,3,4], [1,2,3,4,5]):
    plt.scatter(X_rp[yyy == i, 0], X_rp[yyy == i, 1], color=color, lw=0.1, label=target_name)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.xlabel("RP1")
plt.ylabel("RP2")
plt.show()

X_clust = X_rp
#endregion



# %% region[blue] AUTOENCODER
encoding_dims = [1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,60,70,80,100]
sample_dim = XXX.shape[1]

min_training_losses = []
avg_training_losses = []
max_training_losses = []
min_testing_losses = []
avg_testing_losses = []
max_testing_losses = []
for encoding_dim in encoding_dims:
    print("╔═════════════════════════╗")
    print("║ ENCODING DIMENSION = {:<2} ║".format(encoding_dim))
    print("╚═════════════════════════╝")
    
    training_losses = []
    testing_losses = []

    ticksize, tickmarks = tickinit(nbr_runs)
    chrono()
    for run_nbr in range(nbr_runs):
        input_img = Input(shape=(sample_dim,))
        # encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(sample_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        
        history = autoencoder.fit(X_train, X_train, verbose=False, epochs=200, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

        training_losses.append(history.history['loss'][-1])
        testing_losses.append(history.history['val_loss'][-1])
        tick(ticksize, tickmarks, run_nbr)
    print("│  {0:.2f}s\n".format(chrono()))

    min_training_losses.append(np.amin(training_losses))
    avg_training_losses.append(sum(training_losses)/len(training_losses))
    max_training_losses.append(np.amax(training_losses))

    min_testing_losses.append(np.amin(testing_losses))
    avg_testing_losses.append(sum(testing_losses)/len(testing_losses))
    max_testing_losses.append(np.amax(testing_losses))


fig, ax = plt.subplots()
ax.plot(encoding_dims,avg_training_losses, label="Training loss", color='IndianRed', linewidth=3.0)
plt.fill_between(encoding_dims, min_training_losses, max_training_losses, color='IndianRed', alpha=0.15)
ax.plot(encoding_dims,avg_testing_losses, label="Testing loss", color='SteelBlue', linewidth=3.0)
plt.fill_between(encoding_dims, min_testing_losses, max_testing_losses, color='SteelBlue', alpha=0.15)
ax.set_xlabel('Number of dimensions')
ax.set_ylabel('Loss')
ax.legend()
plt.show()
#endregion



# %% region[blue] AUTOENCODER
encoding_dim = 20

input_img = Input(shape=(sample_dim,))
# encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(sample_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

history = autoencoder.fit(X_train, X_train, verbose=False, epochs=200, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

encoded_samples = encoder.predict(XXX)
decoded_samples = decoder.predict(encoded_samples)

X_clust = encoded_samples
#endregion



# %% region[blue] DEEP AUTOENCODER
encoding_dim = 30

input_img = Input(shape=(sample_dim,))
# encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
encoded = Dense(450, activation='relu')(input_img)
encoded = Dense(150, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(150, activation='relu')(encoded)
decoded = Dense(450, activation='relu')(decoded)
decoded = Dense(sample_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

history = autoencoder.fit(X_train, X_train, verbose=False, epochs=200, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

encoded_samples = encoder.predict(XXX)
decoded_samples = decoder.predict(encoded_samples)

X_clust = encoded_samples
#endregion



#%% region[yellow] K-MEANS
km = KMeans(n_clusters=10, random_state=0).fit(X_clust)
labs = np.array([km.labels_]).transpose()
scaler = preprocessing.StandardScaler()
labs = scaler.fit_transform(labs)
X_clust = np.concatenate((X_clust,labs),axis=1)
#endregion



#%% region[green] EXPECTATION MAXIMIZATION
km = KMeans(n_clusters=10, random_state=0).fit(X_clust)
labs = np.array([km.labels_]).transpose()
scaler = preprocessing.StandardScaler()
labs = scaler.fit_transform(labs)
X_clust = np.concatenate((X_clust,labs),axis=1)
#endregion



# %% region[black] NEURAL NETWORK
X_clust_train, X_clust_test, y_clust_train, y_clust_test = train_test_split(X_clust, yyy, test_size=0.1, random_state=0)
network = (12,12,12)
print("For the network {}".format(network))
nn = MLPClassifier(hidden_layer_sizes=network, activation='relu', verbose=False, max_iter=4000)
nn = analyse(nn, fold, XXX, yyy, X_clust_train, y_clust_train, X_clust_test, y_clust_test)
winsound.Beep(frequency, duration)
#endregion





# %%

