PCA file transforms the original set of potentially correlated variables into a new set of uncorrelated variables (principal components), 
ordered by the amount of variance they explain. 
The "essential" variables in the context of PCA are the top principal components that capture a significant portion of the data's variability, 
effectively reducing the dimensionality while preserving crucial information. 
The loadings of the original features within these principal components can further indicate which original features are most 
influential in driving these essential components.

The 2PCA code demonstrates how PCA can be used to reduce the dimensionality of the breast cancer dataset down to two essential components, 
allowing for visualization and potentially simplifying further analysis while retaining a significant portion of the original variance. 
The analysis of component loadings helps in understanding which original features contribute most to these new, lower-dimensional representations.

Logistic regression is a linear model used for binary classification problems (where the outcome can be one of two classes). It works by:

Calculating a weighted sum of the input features: z = b0 + b1*x1 + b2*x2 + ... + bn*xn, where bi are the coefficients and xi are the features.
Applying the sigmoid function: The result z is then passed through the sigmoid function (also known as the logistic function): p = 1 / (1 + e^-z). The sigmoid function squashes the output to a probability between 0 and 1.
Making a prediction: A threshold (usually 0.5) is used to classify the instance. If the predicted probability p is greater than the threshold, the instance is classified as one class (e.g., malignant), otherwise as the other class (e.g., benign).
The model learns the optimal coefficients during the training process by trying to minimize a cost function (like cross-entropy loss) that measures the difference between the predicted probabilities and the actual labels.
