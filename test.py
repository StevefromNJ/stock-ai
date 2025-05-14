import bentoml

sma_runner = bentoml.sklearn.get('sma_10_prediction:latest').to_runner()
sma_runner.init_local()
print(sma_runner.predict.run([[5110.47, 5137.45]]))