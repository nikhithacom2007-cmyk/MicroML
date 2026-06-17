from microml.metrics.metrics import *

y_true = [1, 2, 3, 4]
y_pred = [1.1, 2.2, 2.8, 3.9]

print("MSE :", mse(y_true, y_pred))
print("MAE :", mae(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))

print(
    "Accuracy:",
    accuracy(
        [0, 1, 1, 0],
        [0, 1, 0, 0]
    )
)