#!/usr/bin/env python
'''Introduction to Machine Learning'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

def classify():
    '''Build a classifier for iris dataset'''
    # import dataset
    iris = load_iris()

    # Show the dataset features and targets
    print("Dataset features:")
    print(iris.feature_names)
    print("Dataset targets :")
    print(iris.target_names)

    # Show full dataset
    for i in range(len(iris.target)):
        print("%d: features %s, target %s" % (i, iris.data[i], iris.target[i]))

    # Choose several arbitrary row indices as test data
    test_index = [1, 53, 101]

    # Assign training data
    training_data = np.delete(iris.data, test_index, axis=0)
    training_target = np.delete(iris.target, test_index)

    # Assign testing data
    testing_data = iris.data[test_index]
    testing_target = iris.target[test_index]

    # Create a classifier
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(training_data, training_target)

    # Testing the classifier using testing data
    print("Testing features:")
    print(testing_data)
    print("Desired Target  :")
    print(testing_target)
    print("Predicted Target:")
    print(classifier.predict(testing_data))

    # Visualizing the decision tree
    dot_data = tree.export_graphviz(
        classifier, out_file=None, feature_names=iris.feature_names,
        class_names=iris.target_names, filled=True, rounded=True,
        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("visualization")


if __name__ == "__main__":
    classify()
