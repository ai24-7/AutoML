# this is my python file to test auto keras capabilities.
from tensorflow.keras.datasets import mnist
import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()
clf = ak.ImageClassifier(max_trials=2,
                        loss='categorical_crossenthropy',
                        metrics=['accuracy'],
                        objective='val_accuracy')
clf.fit(x_train, y_train,
        validation_split=0.15,
        epochs=3,
        verbose=2
        )
test_loss, test_acc = clf.evaluate(x_test, y_test, verbose=0)
print(f'test accuracy is {test_acc} and test loss is {test_loss}')

print('####################' * 4)

best_model = clf.export_model()
best_model.summary()

