# ML_solution
<<<<<<< HEAD
ADC reading algorithm


# Python Tools
=======

ADC reading algorithm

# Python Tools

>>>>>>> tanbranch
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

pip install tensorflow

pip install scikit-learn

<<<<<<< HEAD

# .png filer

mimeopen -d || gio open
=======
# For Trkhush

genWave.py - creates waveforms close to reality in folder waveform_raw

baseline_subtract.py - removes baselines and saves waveforms in folder waveform_baseline_removed

prepare_ml_dataset.py - creates training_data_signals.npz and training_data_counts.npz

train_count_model.py - extracts number of signals

train_signal_model.py - extracts signal (t0,amplitude) pairs

compare_signal_predictions.py - Compares Models predictions against truth values, saves in folder comparison_plots

plot_individual_waveform.py - Plots baseline subtructed waveforms with predictions and truth values by looping over
			      predicted counts from train_count_model.py and using (t0,amplitude) pairs from train_signal_model.py

# open .png files

mimeopen -d || gio open

# delete files

find waveform_raw/ -type f -delete

# git upload

git add .

git commit -m "Added a new feature"

git push origin tanbranch
>>>>>>> tanbranch
