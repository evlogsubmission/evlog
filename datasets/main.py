from .evlog import EVLOG_Dataset


def load_dataset(data_path, train_folder, evolved_folder):


    dataset = EVLOG_Dataset(data_path, train_folder, evolved_folder)


    return dataset
