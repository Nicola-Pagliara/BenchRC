from Preprocessing import preprocessing_DoCRED as prep
from Models import train as trn
import Evaluation.eval as evl
from Support import constant as const

def pipeline():
    # add tail head and genralize it
    #prep.prepare_intra_dataset()
    prep.convert_head_tail_dataset(const.DATASET_TRAIN_HUMAN)
    print('Preprocessing phase ending')
    trn.general_train()
    print('Train phase ending')
    #evl.compute_numeric_metrics()
    #print('Evaluation phase ending')
    return


def main():

    pipeline()

if __name__ == "__main__":
        main()