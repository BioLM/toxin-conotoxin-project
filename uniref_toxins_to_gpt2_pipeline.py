"""Process one of the UniRef toxins files to create inputs and outputs
for training a GPT2 generative model. This essentially entails:

    - Parse files to sequences only
    - Make sure each sequence has only valid AA characters, filter
    - Shuffle sequences
    - Remove anything too short or too long for the model
    - Drop duplicates just in case
    - Make 80/20 train test split TXT files, one seq per line
"""
import os
import pandas as pd

AA = ('A', 'C', 'D', 'E', 'F', 'G', 'H',
      'I', 'K', 'L', 'M', 'N', 'P', 'Q',
      'R', 'S', 'T', 'V', 'W', 'Y', 'U')
AAs = ''.join(AA)

def clean_and_split(fpath):
    """Clean and split one of the input TSV files."""
    df = pd.read_csv(fpath, sep='\t')
    seqs = df['Reference sequence']

    print("Beginning with {} sequences".format(seqs.shape[0]))

    # Remove anything shorter than 8 AA (I believe approx 2 tokens) and too long
    seqs = seqs.loc[(seqs.str.len() >= 8) & (seqs.str.len() <= 510)].reset_index(drop=True)

    # Remove invalid chars
    seqs_desired = seqs.str.match(pat=f'^[{AAs}]+$', case=True)
    seqs = seqs.loc[seqs_desired].reset_index(drop=True)

    # Drop duplicates
    seqs = seqs.drop_duplicates().reset_index(drop=True)

    # Shuffle
    seqs = seqs.sample(n=seqs.shape[0], replace=False)

    # Split in to train and test
    print("Ending with {} sequences".format(seqs.shape[0]))
    split_idx = int(seqs.shape[0] * 0.80)
    train_seqs = seqs.iloc[0:split_idx]
    test_seqs = seqs.iloc[split_idx:]
    print("Train: {}\nTest: {}".format(train_seqs.shape[0], test_seqs.shape[0]))

    def write_txt_seqs(_s, fname):
        """Write out sequence file to same dir as input file."""
        with open(os.path.join(os.path.dirname(fpath), fname), 'w') as f:
            for _seq in _s:
                f.write(f'{_seq}\n')

    write_txt_seqs(train_seqs, 'train_raw.txt')
    write_txt_seqs(test_seqs, 'test_raw.txt')


if __name__ == '__main__':
    """Run this Python file."""
    fname = 'toxins_uniref_length4to500_90level.tsv'
    fpath = os.path.join(os.path.dirname(__file__), 'UniRef', 'UniRef_Toxins', fname)
    clean_and_split(fpath)
