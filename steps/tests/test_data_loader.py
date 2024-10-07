import sys
if "." not in sys.path: sys.path.append(".")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#pipe = Pipeline(["passthrough", "passthrough"])
#print(pipe)
from steps import train_data_preprocessor, data_loader, train_data_splitter

dataset, target, random_state = data_loader(42)
print('data_loader')
print(f'dataset.shape = {dataset.shape}')
print(f'target = {target}')
print(f'random_state = {random_state}')
print('--'*30)

dataset_trn, dataset_tst = train_data_splitter(dataset)
print('train_data_splitter')
print(f'dataset_trn.shape = {dataset_trn.shape}')
print(f'dataset_tst.shape = {dataset_tst.shape}')
print('--'*30)

dataset_trn, dataset_tst, preprocess_pipeline = train_data_preprocessor(
    dataset_trn,
    dataset_tst,
    normalize = True
)
print(f'train_data_preprocessor')
print(f'dataset_trn.shape = {dataset_trn.shape}')
print(f'dataset_tst.shape = {dataset_tst.shape}')
print(f'preprocess_pipeline = {preprocess_pipeline}')