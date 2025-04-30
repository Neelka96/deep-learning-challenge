from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def io_dimensions(data: pd.DataFrame | pd.Series | np.ndarray) -> int:
    if isinstance(data, pd.DataFrame) or (isinstance(data, np.ndarray) and len(data.shape) <= 2):
        dims = data.shape[1]
    elif isinstance(data, pd.Series):
        dims = 1
    else:
        raise TypeError('Bad argument type. Currently only excepts pandas and numpy objects.')
    return dims


class Preprocessor:
    def __init__(self, uri: str):
        self.df = pd.read_csv(uri)
        self.safe_copy = self.df.copy()
        self.help = \
        '''
            Welcome to the preprocessor. 
            Steps are:
                - Construct (Done)
                - Bin features with high unique value counts with `.binning()`
                - Sanitize data with `.cleanup()`
                - Split, fit, and transform with `.split_data()`
        '''
        print(self.help)

    def reset_df(self):
        self.df = self.safe_copy.copy()
        print('Dataframe reset to original.')
        return self
    
    def recommend_binning(self) -> dict:
        suspects = dict()
        for val, count in self.nunique().items():
            if count > 10: suspects[val] = count
        return suspects
    
    def nunique(self, dtype: str | None = None) -> pd.Series | int:
        if dtype is not None:
            selected: pd.Series | pd.DataFrame = self.df.select_dtypes(include = [dtype])
        else:
            selected = self.df
        return selected.nunique()
    
    def __convert_col_num__(self, col: int) -> str:
        return self.df.columns[col]
    
    def get_column(self, col: str | int | None = None) -> str:
        if col is not None:
            try:
                col = col.upper()
            except:
                col = self.__convert_col_num__(col)
            else:
                raise TypeError('Incompatible type given for column parameter.')
            finally:
                return col
        else:
            return self.df.columns
    
    def get_series(self, col: str) -> pd.Series:
        try:
            series = self.df[col]
        except:
            series = self.df[self.get_column(col)]
        else:
            raise ValueError(f'Column {col} could not be found')
        finally:
            return series

    def value_counts(self, col) -> pd.Series:
        return self.get_series(col).value_counts()
    
    def info(self) -> None:
        return self.df.info()

    def binning(self, col: str, freq_min: int, overwrite = True) -> pd.Series:
        try:
            # Try to convert numerical column to string if neccessary
            col = self.get_column(col)

            # Get value counts and iterate through them checking against minimum neccessary frequency
            vals_to_repl = [val for val, count in self.value_counts(col).items() if count < freq_min]

            # Replace in dataframe
            for val in vals_to_repl:
                self.df[col] = self.df[col].replace(val, 'Other')
        except:
            if overwrite:
                self.reset_df()
                self.binning(col, freq_min, overwrite = False)
            else: raise RuntimeError('Overwriting turned off for binning method. Exiting early.')
        finally:
            # Check to make sure replacement was successful
            return self.value_counts(col)


    def cleanup(self, drop_ids: list[str] | None = None):
        # If drop_ids is passed, drop them inplace from df
        if drop_ids is not None:
            drop_ids = [self.get_column(id) for id in drop_ids]
            self.df.drop(columns = drop_ids, inplace = True)

        # Convert all object columns into dummies and concat them
        # Create temporary df and list for categorical data
        self.original_cols = self.df.columns
        category_df = self.df.select_dtypes(include = ['object'])
        categories = list(category_df.columns)
        self.df = pd.concat([self.df, pd.get_dummies(category_df)], axis = 1)
        self.df.drop(columns = categories, inplace = True)
        print('Cleanup successful!')
        return self
    
    def split_data(self, target: str | int, **kwargs):
        target = self.get_column(target)

        # Separate out the target from the features
        y = self.df[target]
        X = self.df.drop(columns = [target])

        # Split training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, **kwargs)

        # Scale data
        scaler = StandardScaler()
        X_scaler = scaler.fit(X_train)

        # Transform and save scaled results for X matrices
        X_train_scaled = X_scaler.transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        
        # Get io dimensions for later calculations
        input_dims = io_dimensions(X_train_scaled)
        output_dims = io_dimensions(y_train)

        return {
            'original_columns': self.original_cols
            ,'input_dims': input_dims
            ,'output_dims': output_dims
            ,'X_train': X_train
            ,'X_train_scaled': X_train_scaled
            ,'X_test': X_test
            ,'X_test_scaled': X_test_scaled
            ,'y_train': y_train
            ,'y_test': y_test
        }


# EOF

if __name__ == '__main__':
    print('Sorry, this module is for import only, not direct execution.')