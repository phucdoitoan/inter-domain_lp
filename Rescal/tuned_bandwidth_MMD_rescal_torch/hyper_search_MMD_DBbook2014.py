

import torch
from time import time
import pickle

import optuna
from hyper_search_MMD import objective, data_file_dict



N_TRIALS = 10 #20 #50

with torch.cuda.device(2):

	data_name = 'DBbook2014'

	skip_list = []
	print('skip_list is: ', skip_list)

	for data_file, overlap in data_file_dict[data_name]:
		if overlap in skip_list:
			print('************* skip overlap = %s ***********' %overlap)
			continue

		t0 = time()
		study = optuna.create_study(direction='maximize')
		study.optimize(lambda trial: objective(trial, data_file, data_name, overlap), n_trials=N_TRIALS)

		pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
		complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

		trial = study.best_trial

		with open('best_hypers/%s_overlap=%s_best_hypers_MMD.pkl' %(data_name, overlap), 'wb') as file:
			pickle.dump(trial, file)

		with open('best_hypers/%s_overlap=%s_best_hypers_MMD.txt' %(data_name, overlap), 'w') as file:

			file.write("\nStudy statistics: ")
			file.write("\n  Number of finished trials: %s" %len(study.trials))
			file.write("\n  Number of pruned trials: %s" %len(pruned_trials))
			file.write("\n  Number of complete trials: %s" %len(complete_trials))

			file.write('\n\n\nBest hypers of %s with overlap=%s\n || Model = MMD' %(data_name, overlap))
			file.write('\nBest Hit@10: %.5f\n' %(trial.value))

			file.write('\nHyperparametsr:\n')

			for key, value in trial.params.items():
				file.write('\t%s : %s\n' %(key, value))

			file.write('Total time for hyperparameter searching: %.2fs' %(time() - t0))



