import datasets

def download_cybergym_subset():
    # Download the CyberGym subset from Hugging Face
    dataset = datasets.load_dataset('sunblaze-ucb/cybergym')
    # Save a small sample for local evaluation
    sample = dataset['tasks'].select(range(10))
    sample.to_json('cybergym_subset_sample.json')
    print('Sample saved to cybergym_subset_sample.json')

if __name__ == '__main__':
    download_cybergym_subset()
