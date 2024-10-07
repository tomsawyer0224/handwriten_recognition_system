# Apache Software License 2.0
# 
# Copyright (c) Tom Sawyer 2024. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 


from typing import List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Annotated
from zenml import step

from utils.preprocess import ColumnsDropper, DataFrameCaster, NADropper


@step
def train_data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    drop_na: Optional[bool] = None,
    normalize: Optional[bool] = None,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """Data preprocessor step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model training. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps like cleaning,
    feature engineering, feature selection, etc. It then returns the processed
    dataset as an step output artifact.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to drop NA values, drop some
    columns and normalize numerical columns. See the documentation for more
    information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        drop_na: If `True` all NA rows will be dropped.
        normalize: If `True` all numeric fields will be normalized.
        drop_columns: List of column names to drop.

    Returns:
        The processed datasets (dataset_trn, dataset_tst) and fitted `Pipeline` object.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
    if drop_na:
        preprocess_pipeline.steps.append(("drop_na", NADropper()))
    if drop_columns:
        # Drop columns
        preprocess_pipeline.steps.append(("drop_columns", ColumnsDropper(drop_columns)))
    if normalize:
        # Normalize the data
        preprocess_pipeline.steps.append(("normalize", MinMaxScaler()))
    #preprocess_pipeline.steps.append(("cast", DataFrameCaster(dataset_trn.columns)))
    
    target_trn =  dataset_trn['target']
    target_tst = dataset_tst['target']
    dataset_trn_wo_target = dataset_trn.drop(columns=['target'])
    dataset_tst_wo_target = dataset_tst.drop(columns=['target'])
    preprocess_pipeline.steps.append(("cast", DataFrameCaster(dataset_trn_wo_target.columns)))
    dataset_trn_wo_target = preprocess_pipeline.fit_transform(dataset_trn_wo_target)
    dataset_tst_wo_target = preprocess_pipeline.transform(dataset_tst_wo_target)
    #dataset_trn_wo_target = preprocess_pipeline.fit_transform(dataset_trn.drop(['target'], axis=1))
    #dataset_tst_wo_target = preprocess_pipeline.transform(dataset_tst.drop(['target'], axis=1))
    
    #dataset_trn = preprocess_pipeline.fit_transform(dataset_trn)
    #dataset_tst = preprocess_pipeline.transform(dataset_tst)
    
    #print(f'dataset_trn_wo_target.shape = {dataset_trn_wo_target.shape}')
    #print(f'target_trn.shape = {target_trn.shape}')

    #print(f'dataset_tst_wo_target.shape = {dataset_tst_wo_target.shape}')
    #print(f'target_tst.shape = {target_tst.shape}')

    dataset_trn = pd.concat([dataset_trn_wo_target, target_trn], axis=1)
    dataset_tst = pd.concat([dataset_tst_wo_target, target_tst], axis=1)

    #print(f'dataset_trn.shape = {dataset_trn.shape}')
    #print(f'dataset_tst.shape = {dataset_tst.shape}')
    ### YOUR CODE ENDS HERE ###

    return dataset_trn, dataset_tst, preprocess_pipeline
