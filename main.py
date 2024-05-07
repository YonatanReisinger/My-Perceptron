import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self):
        self.current_weights = None
        self.current_constant = 0
        self.fit_progress_completed = False
        self.data_training_set = None
        self.data_training_true_labels = None
        self.best_fitting_score = 0
        self.encoded_positive_label = 1
        self.encoded_negative_label = -1


    def _select_better_weights(self, new_weights: list, new_constant: int) -> tuple:
        """
        compare the best weights so far to the new ones by comparing the amount of good classifications each weights make
        Args:
            new_weights (list): best weights after one change of a misclassified data point
            new_constant (int): best constant after one change of a misclassified data point
        Returns:
            tuple with the best weights and constant between the two weights
        """
        current_weights = self.current_weights
        current_constant = self.current_constant

        # Try the new weights
        self.current_weights = new_weights
        self.current_constant = new_constant
        new_weights_score = self.score(self.data_training_set, self.data_training_true_labels)

        if(self.best_fitting_score >= new_weights_score): # return the values with higher score
            return current_weights, current_constant
        else:
            self.best_fitting_score = new_weights_score
            return new_weights, new_constant

    def _update_best_weights(self, best_weights: list, best_constant: int, feature_vector, encoded_label: int) -> tuple:
        """
        Updates the weights and constant based on the misclassification error found.
        Args:
            best_weights (list): best weights found so far.
            best_constant (float): best constant found so far.
            feature_vector (list): Feature vector representing a single data point.
            encoded_label (int): Encoded label (-1 or 1).
        Returns:
            Tuple: Updated weights, constant, and a boolean indicating if an error was found.
        """
        feature_vector = feature_vector.tolist()
        inner_product_plus_constant = np.inner(feature_vector, best_weights) + best_constant
        # if a data point was predicted as negative, but it is really a positive data point
        if inner_product_plus_constant < 0 and encoded_label == 1:
            new_weights = np.add(best_weights, feature_vector)
            new_constant = best_constant + 1
            error_found = True
            best_weights, best_constant = self._select_better_weights(new_weights,new_constant)

        # if a data point was predicted as positive, but it is really a negative data point
        elif inner_product_plus_constant >= 0 and encoded_label == -1:
            new_weights = np.subtract(best_weights, feature_vector)
            new_constant = best_constant - 1
            error_found = True
            best_weights, best_constant = self._select_better_weights(new_weights,new_constant)
        else:
            error_found = False

        return best_weights, best_constant, error_found
    def _predict_all_data_and_update_weights(self, encoded_labels: list) -> bool:
        """
        Predicts labels for all data points and updates weights based on classification errors.
        Args:
            encoded_labels (list): Encoded true labels (-1 or 1).
        Returns:
            a boolean indicating if there were any false predictions.
        """
        best_weights = self.current_weights
        best_constant = self.current_constant
        has_false_predictions = False
        for label_index, feature_vector in self.data_training_set.iterrows():
            best_weights, best_constant, error_found = self._update_best_weights(best_weights, best_constant,
                                                                         feature_vector,
                                                                         encoded_labels[label_index])
            if error_found:
                has_false_predictions = True

        self.current_weights = best_weights
        self.current_constant = best_constant
        return has_false_predictions

    def _encode_true_labels(self, true_labels: list) -> list:
        """
        Converts original labels to -1 and 1.
       Args:
            true_labels (list): Original labels.
        Returns:
           list: Encoded labels (-1 or 1).
        """
        # Convert the true labels to 1s and -1s
        positive_label = list(set(true_labels))[0]
        true_labels = np.array(true_labels)
        # encode each label that is considered to be the positive label with 1 and -1 otherwise
        encoded_labels = np.where(true_labels == positive_label, self.encoded_positive_label, self.encoded_negative_label)
        return list(encoded_labels)

    def fit(self, X, y, max_iterations = 100):
        """
        Fits the perceptron model to the given training data.
        Args:
            X (DataFrame or matrix): Feature vectors.
            y (list): True labels.
            max_iterations (int): Maximum number of iterations.
        Returns:
            None
        """
        if len(set(y)) != 2:
            raise RuntimeError ("Perceptron can only do binary classification")

        self.data_training_set = pd.DataFrame(X)
        self.data_training_true_labels = y
        self.fit_progress_completed = True
        # Initialize the weights with 0
        self.current_weights = [0 for _ in range(len(self.data_training_set.iloc[0]))]
        encoded_labels = self._encode_true_labels(y)
        has_false_predictions  = True
        iteration_number = 0

        while has_false_predictions  and iteration_number < max_iterations:
            has_false_predictions = self._predict_all_data_and_update_weights(encoded_labels)
            iteration_number += 1

    def _compute_predictions(self, X: pd.DataFrame) -> list:
        """
        Calculates predictions for all data points using the current weights and constant.
        Args:
            X (DataFrame): Feature vectors.
        Returns:
            list: Predictions for each data point.
        """
        # calculate the inner product of each vector in X with the current weights
        predictions = X.apply(lambda feature_vector:  np.inner(list(feature_vector), self.current_weights) + self.current_constant, axis = 1)
        return predictions
    def predict(self, X) -> list:
        """
        Predicts labels for the given data points using the fitted model.
        Args:
            X (DataFrame or matrix): Feature vectors.
        Returns:
            list: Predicted labels (-1 or 1).
        """
        X = pd.DataFrame(X) # Convert X to a data frame
        if self.fit_progress_completed:
            predictions = self._compute_predictions(X)
            predictions = np.array(predictions)
            # Predict all the vectors of above the line to have a positive prediction and negative otherwise
            predicted_labels = list(np.where(predictions >= 0, self.encoded_positive_label, self.encoded_negative_label))
            return predicted_labels
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def _calculate_num_of_correct_labels(self, data_set, true_labels) -> int:
        """
        Calculates the number of correctly classified labels based on predictions and true labels.
        Args:
            data_set (DataFrame or matrix): Feature vectors.
            true_labels (list): True labels.
        Returns:
            int: Number of correctly classified labels.
        """
        predicted_labels = self.predict(data_set)
        encoded_labels = self._encode_true_labels(true_labels)

        predicted_labels_np = np.array(predicted_labels)
        encoded_labels_np = np.array(encoded_labels)

        return np.sum(predicted_labels_np == encoded_labels_np)


    def score(self, X, y) -> float:
        """
        Calculates the accuracy score of the model based on predictions and true labels.
        Args:
            X (DataFrame or matrix): Feature vectors.
            y (list): True labels.
        Returns:
            float: Accuracy score of the model.
        """
        num_of_samples = len(pd.DataFrame(X))
        num_of_correct_labels = self._calculate_num_of_correct_labels(X,y)
        return float(num_of_correct_labels) / float(num_of_samples)



def main1():
    data_set = [[-2, -1], [0,0], [2,1], [1,2], [-2,2], [-3,0]]
    labels = [-1,1,1,1,-1,-1]
    perceptron = Perceptron()
    perceptron.fit(data_set, labels)
    print(f"The perceptron found the weights vector of {perceptron.current_weights}, constant of: {perceptron.current_constant} and achieved an error of {1 - perceptron.score(data_set, labels)}")

def main2():
    data_set = pd.read_csv("Processed Wisconsin Diagnostic Breast Cancer.csv")
    data_set = data_set.sample(frac = 1).reset_index(drop = True) # Shuffle the data
    labels = data_set["diagnosis"]
    data_set = data_set.drop(columns= ["diagnosis"])

    # 80% of that data will be for training
    number_of_train_samples = int(0.8 * len(data_set))
    data_set_train = data_set.iloc[: number_of_train_samples]
    data_set_test = data_set.iloc[number_of_train_samples : ]
    labels_train = labels.iloc[: number_of_train_samples]
    labels_test = labels.iloc[number_of_train_samples : ]

    perceptron = Perceptron()
    perceptron.fit(data_set_train, labels_train)
    print(f"The error on the training data set is: {1 - perceptron.score(data_set_train, labels_train)}"
          f" and error on the testing data set is: {1 - perceptron.score(data_set_test, labels_test)}")


if __name__ == '__main__':
    print("This is the first main (Question D):")
    main1()
    print("\nThis is the second main (Question E):")
    main2()