

import torch
from create_warmstart_embeddings import main


if __name__ == '__main__':
	with torch.cuda.device(2):
		data_file_dict = {
			'FB15k-237': [('../data/FB15k-237/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/FB15k-237/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/FB15k-237/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/FB15k-237/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
			'WN18RR': [('../data/WN18RR/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/WN18RR/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/WN18RR/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/WN18RR/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
			'DBbook2014': [('../data/KG_datasets/dbbook2014/kg/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/KG_datasets/dbbook2014/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/KG_datasets/dbbook2014/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/KG_datasets/dbbook2014/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
			'ML1M': [('../data/KG_datasets/ml1m/kg/divided/fix_intra-test/kg_valid_dict.pkl', 0.0),
						  ('../data/KG_datasets/ml1m/semi/divided/fix_intra-test/kg_valid_dict.pkl', 1.5),
						  ('../data/KG_datasets/ml1m/semi_3/divided/fix_intra-test/kg_valid_dict.pkl', 3.0),
						  ('../data/KG_datasets/ml1m/semi_5/divided/fix_intra-test/kg_valid_dict.pkl', 5.0), ],
		}

		for data_name in ['DBbook2014']:
			for data_file, overlap in data_file_dict[data_name]:
				for emb_dim in [100]:
					for repeat in range(5): #10
						batch_size = 500
						learning_rate = 0.005

						main(data_file, data_name, emb_dim, batch_size, learning_rate, overlap, repeat)