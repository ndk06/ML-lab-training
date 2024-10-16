import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('data/x28.txt', header=None, sep=r'\s+', index_col=0)
    df.columns = [f'A{i}' for i in range(1, 16)] + ['B']
    df = df.reset_index(drop=True)

    return df.iloc[:, :-1], df.iloc[:, -1]


def normalize_and_add_ones(X):
    X = np.array(X)

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.ones(X_normalized.shape[0]) 

    return np.column_stack((ones, X_normalized))


class RidgeRegression:
    def __init__(self) -> None:
        return
    
    def fit(self, X_train, y_train, LAMBDA):
        assert len(X_train.shape)==2 and X_train.shape[0]==y_train.shape[0]
        W = np.linalg.inv(X_train.T @ X_train + LAMBDA * np.eye(X_train.shape[1])) @ X_train.T @ y_train

        return W
    
    def fit_gradient_descent(self, X_train, y_train, LAMBDA, learning_rate, max_num_epoch=100, batch_size=128):
        W = np.random.randn(X_train.shape[1])
        last_loss = 10e+8

        for ep in range(max_num_epoch):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            y_train = y_train[arr]
            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
            for i in range(total_minibatch):
                index = i * batch_size

                X_train_sub = X_train[index:index + batch_size]
                y_train_sub = y_train[index:index + batch_size]
                grad = X_train_sub.T @ (X_train_sub @ W - y_train_sub) + LAMBDA * W

                W -= learning_rate * grad

            new_loss = self.compute_RSS(self.predict(W, X_train), y_train)
            if (np.abs(new_loss - last_loss) <= 1e-5):
                break
            last_loss = new_loss
        
        return W

    def predict(self, W, X_new):
        X_new = np.array(X_new)
        y_new = X_new @ W
        return y_new
    
    def compute_RSS(self, y_new, y_predicted):
        loss = 1. / y_new.shape[0] * np.sum((y_new - y_predicted) ** 2)

        return loss
    
    def get_the_best_LAMBDA(self, X_train, y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predicted = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predicted)
            return aver_RSS / num_folds


        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(5, current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS
        
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0, minimum_RSS=10000 ** 2, LAMBDA_values=range(50))

        LAMBDA_values = [k * 1. / 1000 for k in range(max(0, (best_LAMBDA-1) * 1000), (best_LAMBDA+1) * 1000)] # step_size = 0.001
        
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS, LAMBDA_values=LAMBDA_values)
        
        return best_LAMBDA
    

if __name__ == '__main__':
    X, y = get_data()

    X = normalize_and_add_ones(X)
    y = (y - y.min()) / (y.max() - y.min())

    X_train = X[:50]
    y_train = y[:50]
    X_test = X[50:]
    y_test = y[50:]

    ridge_regression = RidgeRegression()

    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, y_train)

    W_learned = ridge_regression.fit(X_train, y_train, best_LAMBDA)
    y_predicted = ridge_regression.predict(W_learned, X_test)

    with open('result.txt', 'w') as f:
        f.write(f'Best LAMBDA: {best_LAMBDA}\n')
        f.write(f'RSS: {ridge_regression.compute_RSS(y_test, y_predicted)}')