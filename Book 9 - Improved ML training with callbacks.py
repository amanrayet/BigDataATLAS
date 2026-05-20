import uproot
import pandas as pd
from TLorentzVector import TLorentzVector
import numpy as np
import matplotlib.pyplot as plt
import time
from FileRetriever import print_files as print_files




higgs = uproot.open("http://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/4lep/MC/mc_345060.ggH125_ZZ4lep.4lep.root")
higgsTree = higgs["mini"]

numDataEntries = len(higgsTree["runNumber"].array())
print("Tree contains", numDataEntries, "entries")




bkg = uproot.open("http://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/4lep/MC/mc_363490.llll.4lep.root")
bkgTree = bkg["mini"]

numMCEntries = len(bkgTree["runNumber"].array())

print("Tree contains", numMCEntries, "entries")




def mcWeights(data,lumi=10):
    """
    When MC simulation is compared to data the contribution of each simulated event needs to be
    scaled ('reweighted') to account for differences in how some objects behave in simulation
    vs in data, as well as the fact that there are different numbers of events in the MC tree than 
    in the data tree.
    
    Parameters
    ----------
    data : dataframe containing the data extracted from the TTree
    """

    XSection = data["XSection"]
    SumWeights = data["SumWeights"]
    #These values don't change from event to event
    norm = lumi*(XSection*1000)/SumWeights
    
    scaleFactor_ELE = data["scaleFactor_ELE"]
    scaleFactor_MUON = data["scaleFactor_MUON"]
    scaleFactor_LepTRIGGER = data["scaleFactor_LepTRIGGER"]
    scaleFactor_PILEUP = data["scaleFactor_PILEUP"]
    mcWeight = data["mcWeight"]
    #These values do change from event to event
    scale_factors = scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER*scaleFactor_PILEUP*mcWeight
    
    weight = norm*scale_factors
    return weight




# A cut on lepton charge
def cut_lep_charge(lep_charge):
    """
    Throw away events where the sum of lepton charges is not equal to 0. The first lepton is [0], 2nd lepton is [1] etc.
    
    Parameters
    ----------
    lep_charge : a list containing the charges of the 4 leptons
    """
    return lep_charge[0] + lep_charge[1] + lep_charge[2] + lep_charge[3] != 0

# Cut on lepton type
def cut_lep_type(lep_type):
    """
    Throw away events where we don't have any of: eeee, mumumumu, eemumu
    
    Electron (and positron) lep_type is 11
    Muon (and anti-muon) lep_type is 13

    Parameters
    ----------
    lep_type : a list containing the types of the 4 leptons
    """
    sum_lep_type = lep_type[0] + lep_type[1] + lep_type[2] + lep_type[3]
    
    return (sum_lep_type != 44) and (sum_lep_type != 48) and (sum_lep_type != 52)




def compute_m4l(data):
    """
    A function to calculate the invariant mass of the four-lepton system using each lepton's momentum, direction and energy.

    Parameters
    ----------
    data : a dataframe containing the kinematics of the 4 leptons
    """
    px_total = np.zeros(len(data)) 
    py_total = np.zeros(len(data))
    pz_total = np.zeros(len(data))
    E_total = np.zeros(len(data))

    for i in range(1, 5): # We loop over each lepton in the event. (Note however, pt is a column of the table data which has one event per row)
        pt = data[f'lep_pt_{i}'].values
        eta = data[f'lep_eta_{i}'].values
        phi = data[f'lep_phi_{i}'].values
        E = data[f'lep_E_{i}'].values
        
        px = pt * np.cos(phi) # Therefore, we calculate px for every event in the column with this single command - no for looping over events required!
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)

        px_total += px
        py_total += py
        pz_total += pz
        E_total += E

    m4l_squared = E_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    m4l_squared = np.where(m4l_squared < 0, 0, m4l_squared) # This avoids square rooting a negative (in case Python confuses a small number with a negative)
    m4l = np.sqrt(m4l_squared)
    return m4l




def prepareData(tree, sample, fraction):
    import awkward as ak
    numevents = tree.num_entries

    arrays = tree.arrays(['lep_charge', 'lep_type', 'lep_pt', 'lep_n',
                          'lep_eta', 'lep_phi', 'lep_E', 'jet_n', 'jet_pt',
                          'mcWeight', 'scaleFactor_PILEUP',
                          'scaleFactor_ELE', 'scaleFactor_MUON',
                          'scaleFactor_LepTRIGGER', 'XSection', 'SumWeights'],
                         entry_stop=int(numevents * fraction))

    # Scalar columns (one value per event) -> straightforward
    scalar_cols = ['lep_n', 'jet_n', 'mcWeight', 'scaleFactor_PILEUP',
                   'scaleFactor_ELE', 'scaleFactor_MUON',
                   'scaleFactor_LepTRIGGER', 'XSection', 'SumWeights']
    
    # Jagged columns (list per event) -> convert to Python lists, keep as objects
    jagged_cols = ['lep_charge', 'lep_type', 'lep_pt', 'lep_eta', 'lep_phi', 'lep_E', 'jet_pt']

    data = pd.DataFrame()
    for col in scalar_cols:
        data[col] = ak.to_numpy(arrays[col])
    for col in jagged_cols:
        data[col] = [list(x) for x in arrays[col]]  # each row is a Python list

    fail = data['lep_charge'].apply(cut_lep_charge)
    data = data[~fail]
    fail = data['lep_type'].apply(cut_lep_type)
    data = data[~fail]

    if 'data' not in sample:
        data['totalWeight'] = mcWeights(data)
    else:
        data['totalWeight'] = np.full(len(data), 1.0)

    for i in range(1, 5):
        data[f'lep_pt_{i}'] = data['lep_pt'].apply(lambda x: x[i-1])
        data[f'lep_phi_{i}'] = data['lep_phi'].apply(lambda x: x[i-1])
        data[f'lep_eta_{i}'] = data['lep_eta'].apply(lambda x: x[i-1])
        data[f'lep_E_{i}'] = data['lep_E'].apply(lambda x: x[i-1])
        data[f'jet_pt_{i}'] = data['jet_pt'].apply(lambda x: x[i-1] if len(x) >= i else 0)

    data['m4l'] = compute_m4l(data)

    feature_columns = []
    for i in range(1, 5):
        feature_columns += [f'lep_pt_{i}', f'lep_phi_{i}', f'lep_eta_{i}', f'lep_E_{i}', f'jet_pt_{i}']
    feature_columns += ['jet_n', 'm4l']

    features = data[feature_columns].to_numpy()
    label = 1 if sample == 'Higgs' else 0
    labels = np.full(len(features), label)

    return features, data['totalWeight'].to_numpy(), labels, feature_columns




#Prepare our signal and bakground data using the functions we've defined above
signal, signal_weights, signal_labels, feature_names = prepareData(higgsTree, 'Higgs', fraction=1)
background, background_weights, background_labels, feature_names = prepareData(bkgTree, 'Bkg', fraction=1)

#Cobine and shuffle our neural net inputs, like we did in Notebook 8
features_combined = np.vstack([signal, background])
weights_combined = np.concatenate([signal_weights, background_weights])
labels_combined = np.concatenate([signal_labels, background_labels])
indices = np.random.permutation(len(features_combined))
features = features_combined[indices]
labels = labels_combined[indices]
weights = weights_combined[indices]




plt.figure(figsize=(14, 15))
for i in range(len(feature_names)):
    combined = np.concatenate([signal[:, i], background[:, i]])
    bins = np.linspace(combined.min(), combined.max(), 100)

    plt.subplot(5, 5, i + 1)

    # Order: background first, then Higgs (so Higgs is on top)
    data = [background[:, i], signal[:, i]]
    plot_weights = [background_weights, signal_weights]
    plot_labels = ['ZZ', 'Higgs']
    colors = ['purple', 'green']

    plt.hist(data, bins=bins, stacked=True, weights=plot_weights, label=plot_labels, color=colors) #Wweights parameter lets us histogram with our MC weights

    plt.title(feature_names[i])
    plt.xlabel(feature_names[i])
    plt.ylabel('Weighted Count')
    plt.xlim(0, combined.max() * 0.3)

plt.tight_layout()
plt.show()




plt.figure()
i = len(feature_names) - 1

# Get min and max from combined data to set consistent bins
combined = np.concatenate([signal[:, i], background[:, i]])
bins = np.linspace(combined.min(), combined.max(), 400)

# Reverse the order: background first, then Higgs
data = [background[:, i], signal[:, i]]
plot_weights = [background_weights, signal_weights]
plot_labels = ['ZZ', 'Higgs']
colors = ['purple', 'green']

plt.hist(data, bins=bins, stacked=True, weights=plot_weights, label=plot_labels, color=colors)
plt.title(feature_names[i])
plt.xlabel(feature_names[i])
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.xlim(0, 5e5)
plt.show()




import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    features_scaled, labels, weights, test_size=0.2, random_state=42
)


# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, min_lr=1e-7, mode="min"),

]

#Define our network architechture
model = models.Sequential([
    layers.Input(shape=(features.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

#Compile your model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#Train your model
history = model.fit(
    X_train, y_train, sample_weight=w_train,
    validation_data=(X_test, y_test, w_test),
    epochs=100,
    batch_size=512,
    verbose=1,
    callbacks=callbacks
)




#Plot the training accuracy vs epoch and the validation accuracy vs epoch on the same plot

# ... your code here

plt.plot(history.history['accuracy'], label='Optimised Training Accuracy')
#plt.plot(history.history['val_accuracy'], label = 'Optimised Validation Accuracy')

plt.xlim(0,100)
plt.ylim(0.7,1)
plt.xlabel('Epoch (becoming more trained -->)')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()




#Plot the Classifier Score Distribution for both the signal and background using the event weights

y_pred_scores = model.predict(X_test).ravel()  # sigmoid output: values between 0 and 1

# Select weights for test set:
signal_score_weights = w_test[y_test == 1]
background_score_weights = w_test[y_test == 0]

# And select corresponding prediction scores:
signal_scores = y_pred_scores[y_test == 1]
background_scores = y_pred_scores[y_test == 0]

# Plot
plt.figure()
plt.hist(signal_scores, bins=50, alpha=0.6, label='Higgs (Signal)', color='green', weights=signal_score_weights)
plt.hist(background_scores, bins=50, alpha=0.6, label='ZZ (Background)', color='purple', weights=background_score_weights)

plt.xlabel("Classifier Output (score)")
plt.ylabel("Normalized Count")
plt.title("Classifier Score Distribution")
plt.legend()
plt.tight_layout()
plt.yscale('log')
plt.show()
