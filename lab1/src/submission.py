import yaml
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from typing import Tuple
from model import BaseModel
import os
import pickle

from utils import (
    TrainConfigR,
    TrainConfigC,
    DataLoader,
    Parameter,
    Loss,
    SGD,
    GD,
    save,
)

# You can add more imports if needed


# 1.1
def data_preprocessing_regression(data_path: str, saved_to_disk: bool = False) -> Dataset:
    r"""Load and preprocess the training data for the regression task.

    Args:
        data_path (str): The path to the training data.If you are using a dataset saved with save_to_disk(), you can use load_from_disk() to load the dataset.

    Returns:
        dataset (Dataset): The preprocessed dataset.
    """
    # 1.1-a
    # Load the dataset. Use load_from_disk() if you are using a dataset saved with save_to_disk()
    if saved_to_disk:
        dataset = load_from_disk(data_path) #now saved_to_disk: bool = False
    else:
        dataset = load_dataset(r"Rosykunai/SGEMM_GPU_performance")
    # Preprocess the dataset
    # Use dataset.to_pandas() to convert the dataset to a pandas DataFrame if you are more comfortable with pandas
    dataset = Dataset.to_pandas(dataset["train"])
    dataset['Run_time'] = np.log10(dataset['Run_time']) #取对数
    dataset = Dataset.from_pandas(dataset) # Convert the pandas DataFrame back to a dataset
    return dataset


def data_split_regression(dataset: Dataset, batch_size: int, shuffle: bool) -> Tuple[DataLoader]:
    r"""Split the dataset and make it ready for training.

    Args:
        dataset (Dataset): The preprocessed dataset.
        batch_size (int): The batch size for training.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        A tuple of DataLoader: You should determine the number of DataLoader according to the number of splits.
    """
    # 1.1-b
    # Split the dataset using dataset.train_test_split() or other methods
    dataset = dataset.train_test_split(train_size=0.8)#dict
    train_dataloder = DataLoader(dataset=dataset["train"],batch_size=batch_size,shuffle=shuffle,train=1)#dataset,8
    test_dataset = dataset["test"].train_test_split(train_size=0.5)#dict
    test_dataloder = DataLoader(dataset=test_dataset["train"],batch_size=batch_size,shuffle=shuffle,train=0)#dataset,1
    validation_dataloder = DataLoader(dataset=test_dataset["test"],batch_size=batch_size,shuffle=shuffle,train=0)#dataset,1
    all_dataset = concatenate_datasets([dataset["test"],dataset["train"]])# validation ,train and test
    all_dataloader = DataLoader(dataset=all_dataset,batch_size=batch_size,shuffle=shuffle,train=1)
    add_dataset = concatenate_datasets([test_dataset["test"],dataset["train"]])#validation and train
    add_dataloader = DataLoader(dataset=add_dataset,batch_size=batch_size,shuffle=shuffle,train=1)
    # return train_dataloder, test_dataloder
    return add_dataloader, test_dataloder
    return all_dataloader,test_dataloder#all dataset i can see


# 1.2
class LinearRegression(BaseModel):
    r"""A simple linear regression model.

    This model takes an input '''shaped as''' [batch_size, in_features] and returns
    an output shaped as [batch_size, out_features].

    For each sample [1, in_features], the model computes the output as:

    .. math::
        y = xW + b

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Example::

        >>> from model import LinearRegression
        >>> # Define the model
        >>> model = LinearRegression(in_features=3, out_features=1)
        >>> # Predict
        >>> x = np.random.randn(10, 3) batch_size=1, in_features=[x1,x2,x3]
        >>> y = model(x)
        >>> # Save the model parameters
        >>> state_dict = model.state_dict()
        >>> save(state_dict, 'model.pkl')
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 1.2-a
        # Look up the definition of BaseModel and Parameter in the utils.py file, and use them to register the parameters
        self._parameters["weight"] = np.array([0.017]*in_features)
        self._parameters["bias"] = np.array([0.99]*out_features)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # 1.2-b
        # Implement the forward pass of the model

        return np.array(np.dot(x,self._parameters["weight"])+self._parameters["bias"])


# 1.3
class MSELoss(Loss):
    r"""Mean squared error loss.

    This loss computes the mean squared error between the predicted and true values.

    Methods:
        __call__: Compute the loss
        backward: Compute the gradients of the loss with respect to the parameters
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        r"""Compute the mean squared error loss.

        Args:
            y_pred: The predicted values
            y_true: The true values

        Returns:
            The mean squared error loss
        """
        # 1.3-a
        # Compute the mean squared error loss. Make sure y_pred and y_true have the same shape
        y_error = y_pred - y_true
        sum = 0
        for y in y_error:
            sum +=y* y
        return sum/len(y_error)

    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        r"""Compute the gradients of the loss with respect to the parameters.

        Args:
            x: The input values [batch_size, in_features][4096,14]
            y_pred: The predicted values [batch_size, out_features][4096,1]
            y_true: The true values [batch_size, out_features][4096,1]

        Returns:
            The gradients of the loss with respect to the parameters, Dict[name, grad]
        """
        # 1.3-b
        # Make sure y_pred and y_true have the same shape
        grad = {}
        grad_w = np.zeros(len(x[0])) 
        sum_w = np.zeros(len(x[0])) 
        grad_b = np.zeros(1)
        sum_b = np.zeros(1)
        for k in range(0,len(x)):#batch'size
            for i in range(0,len(x[k])):#in_feature
                sum_w[i]+=x[k][i]*(y_pred[k]-y_true[k])/len(x)
            sum_b[0]+=(y_pred[k]-y_true[k])/len(x)
        for i in range(0,len(x[0])):
            grad_w[i]=(sum_w[i]/len(x[0]))
        grad_b[0]=sum_b[0]
        grad["weight"]=np.array(grad_w) 
        grad["bias"]=np.array(grad_b)
        # print(grad["weight"])
        return grad
    

# 1.4
class TrainerR:
    r"""Trainer class to train for the regression task.

    Attributes:
        model (BaseModel): The model to be trained
        train_loader (DataLoader): The training data loader
        criterion (Loss): The loss function
        opt (SGD): The optimizer
        cfg (TrainConfigR): The configuration
        results_path (Path): The path to save the results
        step (int): The current optimization step
        train_num_steps (int): The total number of optimization steps
        checkpoint_path (Path): The path to save the model

    Methods:
        train: Train the model
        save_model: Save the model
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        loss: Loss,
        optimizer: SGD,
        config: TrainConfigR,
        results_path: Path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.criterion = loss
        self.opt = optimizer
        self.cfg = config
        self.results_path = results_path
        self.step = 0
        self.train_num_steps = len(self.train_loader) * self.cfg.epochs#batch_num * train_time
        self.checkpoint_path = self.results_path / "model.pkl"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:
                # 1.4-a
                # load data from train_loader and compute the loss                
                feature = next(iter(self.train_loader))
                y_pred = self.model([row[0:-1] for row in feature])
                y_true = np.array([row[-1] for row in feature])
                loss = self.criterion(y_pred=y_pred,y_true=y_true)
                loss_list.append(loss)
                pbar.set_description(f"Loss: {loss:.6f}")
                # Use pbar.set_description() to display current loss in the progress bar

                # Compute the gradients of the loss with respect to the parameters
                # Update the parameters with the gradients
                grad = {}
                grad = self.criterion.backward(x=[row[0:-1] for row in feature],y_pred=y_pred,y_true=y_true)
                self.opt.step(grad)
                self.step += 1
                pbar.update()

        plt.plot(loss_list)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(self.results_path / "loss_list.png")
        self.save_model()

    def save_model(self):
        self.model.eval()
        save(self.model.state_dict(), self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")


# 1.6
def eval_LinearRegression(model: LinearRegression, loader: DataLoader) -> Tuple[float, float]:
    r"""Evaluate the model on the given data.

    Args:
        model (LinearRegression): The model to evaluate.
        loader (DataLoader): The data to evaluate on.

    Returns:
        Tuple[float, float]: The average prediction, relative error.
    """
    model.eval()
    step = 0
    y_pred_sum=0
    y_true_sum=0
    while step < len(loader):
        feature = next(iter(loader))
        y_pred = model([row[0:-1] for row in feature])
        y_true = np.array([row[-1] for row in feature])
        # print(type(y_pred),type(y_true))
        for i in range(0,len(y_pred)):
            y_pred_sum+=y_pred[i]
        for i in range(0,len(y_true)):
            y_true_sum+=y_true
        step+=1
    y_error= np.abs(np.mean(y_pred_sum)-np.mean(y_true_sum))/np.mean(y_true_sum)
    # y_error= (np.abs(y_pred-y_true).mean())/np.mean(y_true)
    # pred = np.array([])
    # target = np.array([])
    # 1.6-a
    # Iterate over the data loader and compute the predictions
    # TODO: Evaluate the model

    # Compute the mean Run_time as Output
    # You can alse compute MSE and relative error
    # TODO: Compute metrics
     print(f"Mean Squared Error: {y_error}")

    # print(mu_target)

    # print(f"Relative Error: {relative_error}")
    return np.mean(y_pred_sum)/len(loader)/len(y_pred),y_error

    return NotImplementedError


# 2.1
def data_preprocessing_classification(data_path: str, mean: float, saved_to_disk: bool = False) -> Dataset:
    r"""Load and preprocess the training data for the classification task.

    Args:
        data_path (str): The path to the training data.If you are using a dataset saved with save_to_disk(), you can use load_from_disk() to load the dataset.
        mean (float): The mean value to classify the data.

    Returns:
        dataset (Dataset): The preprocessed dataset.
    """
    # 2.1-a
    # Load the dataset. Use load_from_disk() if you are using a dataset saved with save_to_disk()
    if saved_to_disk:
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path)
    # Preprocess the dataset
    # Use dataset.to_pandas() to convert the dataset to a pandas DataFrame if you are more comfortable with pandas
    # TODO：You must do something in 'Run_time' column, and you can also do other preprocessing steps
    # dataset = Dataset["train"].to_pandas()
    
    # dataset = dataset['train'].add_column("beta_w",[1]*dataset['train'].__len__())
    dataset = Dataset.to_pandas(dataset["train"])
    dataset["beta_w"]=1
    dataset["label"]=(dataset["Run_time"]>mean).astype(int)
    dataset.describe().T
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.remove_columns("Run_time")
    train_dataset = dataset.train_test_split(0.2,0.8)["train"]#train
    validation_dataset = dataset.train_test_split(0.2,0.8)["test"]
    test_dataset = validation_dataset.train_test_split(0.5,0.5)["train"]#test
    validation_dataset = validation_dataset.train_test_split(0.5,0.5)["test"]#validation
    # return dataset#all
    # return test_dataset#test
    return concatenate_datasets([train_dataset,validation_dataset]) #train


def data_split_classification(dataset: Dataset) -> Tuple[Dataset]:
    r"""Split the dataset and make it ready for training.

    Args:
        dataset (Dataset): The preprocessed dataset.

    Returns:
        A tuple of Dataset: You should determine the number of Dataset according to the number of splits.
    """
    # 2.1-b
    # Split the dataset using dataset.train_test_split() or other methods
    # TODO: Split the dataset
    print(len(dataset))
    train_dataset = dataset.train_test_split(0.2)#train
    test_dataset = train_dataset["test"].train_test_split(0.5)["train"]#test
    validation_dataset = train_dataset["test"].train_test_split(0.5)["test"]#validation
    train_dataset = train_dataset["train"]
    # return train_dataset,validation_dataset#0.8 for training
    # return concatenate_datasets(train_dataset,validation_dataset),test_dataset #after add
    return dataset,test_dataset #all i can see


# 2.2
class LogisticRegression(BaseModel):
    r"""A simple logistic regression model for binary classification.

    This model takes an input shaped as [batch_size, in_features] and returns
    an output shaped as [batch_size, 1].

    For each sample [1, in_features], the model computes the output as:

    .. math::
        y = \sigma(xW + b)

    where :math:`\sigma` is the sigmoid function.

    .. Note::
        The model outputs the probability of the input belonging to class 1.
        You should use a threshold to convert the probability to a class label.

    Args:
        in_features (int): Number of input features.

    Example::

            >>> from model import LogisticRegression
            >>> # Define the model
            >>> model = LogisticRegression(3)
            >>> # Predict
            >>> x = np.random.randn(10, 3)
            >>> y = model(x)
            >>> # Save the model parameters
            >>> state_dict = model.state_dict()
            >>> save(state_dict, 'model.pkl')
    """

    def __init__(self, in_features: int):
        super().__init__()
        # 2.2-a
        # Look up the definition of BaseModel and Parameter in the utils.py file, and use them to register the parameters
        # This time, you should combine the weights and bias into a single parameter
        # TODO: Register the parameters
        self.beta = Parameter(np.random.randn(in_features+1,1)*0.001)

    def predict(self, x: np.ndarray) -> np.ndarray:
        r"""Predict the probability of the input belonging to class 1.

        Args:
            x: The input values [batch_size, in_features]

        Returns:
            The probability of the input belonging to class 1 [batch_size, 1]
        """
        # 2.2-b
        # Implement the forward pass of the model
        # TODO: Implement the forward pass
        # print(x.shape,"\n",self.beta.shape)
        # res = np.array(np.dot(x,self.beta))
        # print('\npred = ',1 / (1 + np.exp(-res)),'\nmean is ',np.mean(1 / (1 + np.exp(-res))),'\nbeta is',self.beta)
        # print(res.shape)
        # exit()
        return 1.0 / (1.0 + np.exp(-1.0*np.dot(x,self.beta)))
    
# 2.3
class BCELoss(Loss):
    r"""Binary cross entropy loss.

    This loss computes the binary cross entropy loss between the predicted and true values.

    Methods:
        __call__: Compute the loss
        backward: Compute the gradients of the loss with respect to the parameters
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        r"""Compute the binary cross entropy loss.

        Args:
            y_pred: The predicted values
            y_true: The true values

        Returns:
            The binary cross entropy loss
        """
        # 2.3-a
        # Compute the binary cross entropy loss. Make sure y_pred and y_true have the same shape
        # TODO: Compute the binary cross entropy loss
        y_pred=np.clip(y_pred,1e-5,1e5)
        # print(y_pred.shape,y_true.shape)
        sum = 1.0*np.mean((np.log(y_pred))*(y_true) + (np.log(1-y_pred))*(1-y_true))
        return sum
    
    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        r"""Compute the gradients of the loss with respect to the parameters.

        Args:
            x: The input values [batch_size, in_features]
            y_pred: The predicted values [batch_size, out_features]
            y_true: The true values [batch_size, out_features]

        Returns:
            The gradients of the loss with respect to the parameters [Dict[name, grad]]
        """
        # 2.3-b
        # Make sure y_pred and y_true have the same shape
        # TODO: Compute the gradients of the loss with respect to the parameters
        # print(x.shape)
        grad_w=np.vstack(((np.sum(y_pred-y_true)/(x.shape[0])),np.dot(x[:,:-1].T,(y_pred-y_true)/x.shape[0])))
        # print(np.dot(x.T,(y_pred-y_true)/x.shape[0]).shape)
        return {'beta':grad_w}
# 2.4
class TrainerC:
    r"""Trainer class to train a model.

    Args:
        model (BaseModel): The model to train
        train_loader (DataLoader): The training data loader
        loss (Loss): The loss function
        optimizer (SGD): The optimizer
        config (dict): The configuration
        results_path (Path): The path to save the results
    """

    def __init__(
        self, model: BaseModel, dataset: np.ndarray, loss: Loss, optimizer: GD, config: TrainConfigC, results_path: Path
    ):
        self.model = model
        self.dataset = dataset
        self.criterion = loss
        self.opt = optimizer
        self.cfg = config
        self.results_path = results_path
        self.step = 0
        self.train_num_steps = self.cfg.steps
        self.checkpoint_path = self.results_path / "model.pkl"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:
                # 2.4-a
                # load data from train_loader and compute the loss
                # TODO: Load data from train_loader and compute the loss
                # print('beta = ',list(self.model.parameters()))
                # Use pbar.set_description() to display current loss in the progress bar
                # 1.4-a
                # load data from train_loader and compute the loss
                feature = self.dataset
                y_pred = np.array(self.model(feature[:,:-1]))
                y_true = feature[:,-1:]
                # print(y_true.shape,y_pred.shape)
                # exit()
                # print("\ny_pred is,",y_pred[0],"y_true is ",y_true[0])
                # for i in range(0,len(y_pred)):
                #     if y_pred[i] >=0.5:
                #         y_pred[i] = 1-1e-15
                #     else:
                #         y_pred[i] = 1e-15
                # print("\nnow y_pred is",y_pred[0],",y_true is ",y_true[0])
                loss = -self.criterion(y_pred=y_pred,y_true=y_true)
                loss_list.append(loss)
                pbar.set_description(f"Loss: {loss:.6f}")
                # Use pbar.set_description() to display current loss in the progress bar
                print(self.step)
                # Compute the gradients of the loss with respect to the parameters
                # Update the parameters with the gradients
                grad = {}
                grad = self.criterion.backward(feature[:,:-1],y_pred=y_pred,y_true=y_true)
                # print("\ngrad is:",grad)
                print(grad['beta'].shape)
                # exit()
                self.opt.step(grad)
                # Compute the gradients of the loss with respect to the parameters
                # Update the parameters with the gradients
                # TODO: Compute gradients and update the parameters

                self.step += 1
                pbar.update()

        with open(self.results_path / "loss_list.txt", "w") as f:
            print(loss_list, file=f)
        plt.plot(loss_list)
        plt.savefig(self.results_path / "loss_list.png")
        self.save_model()

    def save_model(self):
        self.model.eval()
        save(self.model.state_dict(), self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")


# 2.6
def eval_LogisticRegression(model: LogisticRegression, dataset: np.ndarray) -> float:
    r"""Evaluate the model on the given data.

    Args:
        model (LogisticRegression): The model to evaluate.
        dataset (np.ndarray): Test data

    Returns:
        float: The accuracy.
    """
    model.eval()
    correct = 0
    feature = dataset
    y_pred = model(feature[:,:-1])
    y_true = feature[:,-1]
    y_pred = (y_pred>=0.5).astype(int)
    for i in range(0,len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct+=1
    # 2.6-a
    # Iterate over the data and compute the accuracy
    # This time, we use the whole dataset instead of a DataLoader.Don't forget to add a bias term to the input
    # TODO: Evaluate the model
    # print(len(y_pred),len(y_true))
    return correct/len(y_pred)
