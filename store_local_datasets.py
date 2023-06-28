from datasets import load_dataset, concatenate_datasets

rdf_dataset = load_dataset("rdfdial", 'multiwoz', download_mode='force_redownload')
#print(rdf_dataset)
concat_rdf_dataset = concatenate_datasets([rdf_dataset['train'], rdf_dataset['validation'], rdf_dataset['test']])

rdf_ids = set(concat_rdf_dataset['dialogue_id'])
train_val_rdf = concat_rdf_dataset.train_test_split(test_size=0.2)
test_val_rdf = train_val_rdf['test'].train_test_split(test_size=0.5)
rdf_dataset.update({'train': train_val_rdf['train'], 'validation': test_val_rdf['train'], 'test': test_val_rdf['test']})

print(rdf_dataset)

#rdf_dataset.save_to_disk("multiwoz_rdf_data")
#raise SystemExit

non_rdf_dataset = load_dataset('multiwoz-convlab2', 'v2.3')

if ('validation' not in non_rdf_dataset.keys()) and ('test' not in non_rdf_dataset.keys()):
    all_data = non_rdf_dataset['train']
    all_data = all_data.filter(lambda x: x['dialogue_id'] in rdf_ids)
    train_val = all_data.train_test_split(test_size=0.2)
    test_val = train_val['test'].train_test_split(test_size=0.5)
    non_rdf_dataset.update({'train': train_val['train'], 'validation': test_val['train'], 'test': test_val['test']})

concat_non_rdf_dataset = concatenate_datasets([non_rdf_dataset['train'], non_rdf_dataset['validation'], non_rdf_dataset['test']])

#non_rdf_ids = set(concat_non_rdf_dataset['dialogue_id'])

concat_non_rdf_dataset = concat_non_rdf_dataset.filter(lambda x: x['dialogue_id'] in rdf_ids)

train_val_non_rdf = concat_non_rdf_dataset.train_test_split(test_size=0.2)
test_val_non_rdf = train_val_non_rdf['test'].train_test_split(test_size=0.5)
non_rdf_dataset.update({'train': train_val_non_rdf['train'], 'validation': test_val_non_rdf['train'], 'test': test_val_non_rdf['test']})
print()
print()
print(non_rdf_dataset)
