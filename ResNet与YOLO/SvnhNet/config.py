import torch

config = {
    'train_path': '../data/images/images',
    'valid_path': '../data/images/images',
    'train_labels': '../data/images.json',
    'valid_labels': '../data/images.json',
    'new_train_labels': '../data/new_mchar_train.json',
    'new_valid_labels': '../data/new_mchar_val.json',
    'test_path': '../data/mchar_test_a',
    'seed': 512,
    'n_epochs': 50,
    'batch_size': 128,
    'early_stop': 8,
    'save_path': './models2/models8.ckpt',
    'learning_rate': 0.001,
    'sample_path': '../data/sample.csv',
    'submit_name': 'submit_8.csv',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
