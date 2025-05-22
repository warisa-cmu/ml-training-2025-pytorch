import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# This class wraps NumPy arrays (X for features, Y for targets) into a PyTorch Dataset, allowing you to use them with DataLoader for batching, shuffling, etc.

# Method Details
# __init__(self, X, Y)
# - Stores the feature and target arrays.

# __len__(self)
# - Returns the number of samples (assumes Y is shaped [num_samples, ...]).

# __getitem__(self, idx)
# - Fetches the idx-th sample from X and Y.
# - Converts them from NumPy arrays to PyTorch tensors and casts them to float.
# - Returns the feature and target tensors as a tuple.


# Assumptions
# - Both X and Y are NumPy arrays.
# - X is at least 2D, as indicated by the slicing self.X[idx, :].
# - The number of samples in X and Y match along the first dimension


class DatasetPT(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        X_out = torch.from_numpy(self.X[idx, :]).float()
        Y_out = torch.from_numpy(self.Y[idx, :]).float()
        return X_out, Y_out


class DataHandlerPT(Dataset):
    def __init__(self, _X, _Y, scalerX, scalerY):
        self._X = _X
        self._Y = _Y
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None

    def split_and_scale(self, test_size, random_state, val_size=0):
        _X_train, _X_test, _Y_train, _Y_test = train_test_split(
            self._X, self._Y, test_size=test_size, random_state=random_state
        )

        self.scalerX.fit(_X_train)
        self.scalerY.fit(_Y_train)

        if val_size > 0:
            _X_train, _X_val, _Y_train, _Y_val = train_test_split(
                _X_train,
                _Y_train,
                # For example, if you want 80% train, 10% validation, and 10% test:
                # First, split off the test set (10%):
                # Next, split the remaining 90% into train and validation.
                # Since you want 80% train and 10% validation overall, the validation set should be 10/90 = 0.111 of the remaining data.
                test_size=val_size / (1 - test_size),
                random_state=random_state + 100,  # Just make random_state different.
            )
            self.X_val = self.scalerX.transform(_X_val)
            self.Y_val = self.scalerY.transform(_Y_val)

        self.X_train = self.scalerX.transform(_X_train)
        self.X_test = self.scalerX.transform(_X_test)

        self.Y_train = self.scalerY.transform(_Y_train)
        self.Y_test = self.scalerY.transform(_Y_test)

    # This part is different from SKLearn version
    def get_train(self):
        return DatasetPT(X=self.X_train, Y=self.Y_train)

    def get_val(self):
        if self.X_val is None:
            raise Exception("No validation data")
        return DatasetPT(X=self.X_val, Y=self.Y_val)

    def get_test(self):
        return DatasetPT(X=self.X_test, Y=self.Y_test)
