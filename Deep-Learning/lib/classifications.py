from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

class Classification_Helper:
    def __init__(self, ml_data: dict, task: str = 'classification', n_estimators: int = 100, rand = True):
        self.org_cols = ml_data.get('original_columns')
        self.X_train = ml_data.get('X_train')
        self.X_test = ml_data.get('X_test')
        self.y_train = ml_data.get('y_train')
        self.y_test = ml_data.get('y_test')
        self.random_state = 78 if rand else None
        self.compute_feature_importances(task, n_estimators)
        self.pca_fit()

    def compute_feature_importances(self, task: str, n_estimators: int) -> pd.DataFrame:
        '''
        Fits a Random Forest and returns a DataFrame of features sorted by importance.
        
        Parameters:
            - task: 'classification' or 'regression'
        '''
        Model = (RandomForestClassifier if task == 'classification' else RandomForestRegressor)
        rf = Model(n_estimators = n_estimators, random_state = self.random_state)
        rf.fit(self.X_train, self.y_train)
        
        self.imp_df = pd.DataFrame(
            {
                'feature': self.X_train.columns,
                'importance': rf.feature_importances_
            }
        )
        for col in self.org_cols:
            self.imp_df['feature'] = \
                self.imp_df['feature'].str.replace(rf'{col}_.*', col, regex = True)
        
        self.imp_df.groupby('feature').sum('importance').sort_values('importance', ascending=False, inplace = True)
        
        return self
    
    def pca_fit(self):
        self.pca = PCA(n_components = 2, random_state = self.random_state)
        self.comps = self.pca.fit_transform(self.X_train)
        return self

    def plot_pca_2d(self, plot_y = True):
        '''
        Performs PCA (2 components) and makes a scatterplot.
        
        If y is provided, it will color code points by y.
        '''
        
        plt.figure(figsize = (6, 5))
        if plot_y:
            sc = plt.scatter(self.comps[:,0], self.comps[:,1], c = self.y_train, cmap = 'viridis', alpha = 0.7)
            plt.colorbar(sc, label = 'Target')
        else:
            plt.scatter(self.comps[:,0], self.comps[:,1], alpha = 0.7)
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:0.2%} var)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:0.2%} var)')
        plt.title('PCA: First Two Principal Components')
        plt.tight_layout()
        plt.show()
        
        return None

    def show_top_n(self, top_n: int = 10):
        top = self.imp_df.head(top_n).iloc[::-1]
        plt.figure(figsize = (6, (top_n * 0.4)))
        plt.barh(top['feature'], top['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
        return None

    def plot_feature(self, feature: str):
        plt.figure(figsize = (6, 4))
        plt.scatter(self.X_test[feature], self.y_test, alpha = 0.6)
        plt.xlabel(feature)
        plt.ylabel('Target')
        plt.title(f'{feature!r} vs Target')
        plt.tight_layout()
        plt.show()
        return None


# EOF

if __name__ == '__main__':
    print('Sorry, this module is for import only, not direct execution.')