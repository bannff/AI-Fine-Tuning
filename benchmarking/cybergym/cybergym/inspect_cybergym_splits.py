import datasets

def inspect_cybergym_splits():
    dataset = datasets.load_dataset('sunblaze-ucb/cybergym')
    print('Available splits:', dataset.keys())
    for split in dataset.keys():
        print(f'Sample from split {split}:', dataset[split][:2])

if __name__ == '__main__':
    inspect_cybergym_splits()
