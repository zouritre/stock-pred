import time
from datetime import datetime
import autokeras as ak
import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from flask_socketio import emit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# tf.compat.v1.disable_eager_execution

# sys.exit()
def getpredictions(datasetname, retrained, forecast, dataToPredict, timeframe, **kwargs):
    data = pd.read_csv(datasetname, skiprows=1)
    data = data[::-1]
    # all_features = data[['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT']]
    dates = data['date']
    dates = dates.to_numpy()
    dates = np.reshape(dates, (-1,))

    allVals = data[dataToPredict].to_numpy()

    closes = data[[dataToPredict]].to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    all_features = scaler.fit_transform(closes)
    newest_feature = all_features[-1]

    y = data[dataToPredict].to_numpy()
    training_sample = len(all_features) * 0.90
    training_sample = round(training_sample)
    try:
        if kwargs['getdates']:
            def minutes(i):
                relDelta = parse(str(dates[-1])) + relativedelta(minutes=+i)
                return relDelta

            def hours(i):
                relDelta = parse(str(dates[-1])) + relativedelta(hours=+i)
                return relDelta

            def days(i):
                relDelta = parse(str(dates[-1])) + relativedelta(days=+i)
                return relDelta

            def forcasteddates(argument, i):
                switcher = {
                    "minutes": minutes(i),
                    "hours": hours(i),
                    "days": days(i)
                }
                func = switcher.get(argument, lambda: "Invalid timeframe")
                return func

            future = []
            for i in range(1, forecast + 1):
                relDelta = forcasteddates(timeframe, i)
                new_now = datetime.strftime(relDelta, '%Y-%m-%d %H:%M:%S')
                future = np.append(future, new_now)

            dates = np.append(dates, future)
            return dates.tolist()
    except Exception:
        if retrained:
            det, strai = retrain_model(retrained, forecast, training_sample, all_features, y, dataToPredict, scaler,
                                       timeframe)
            vals = np.append(allVals, det)
            return vals.tolist(), strai
        else:
            predDetailled = try_load_model(dataToPredict, retrained, forecast, training_sample, all_features, y, scaler,
                                           timeframe, detailled=True)
            predStraight = try_load_model(dataToPredict, retrained, forecast, training_sample, all_features, y, scaler,
                                          timeframe, detailled=False)
            vals = np.append(allVals, predDetailled)
            return vals.tolist(), predStraight


def try_load_model(dataToPredict, retrained, forecast, training_sample, allfeatures, y, scaler, timeframe, detailled):
    detailledSecours = True
    try:
        if detailled:
            detailledSecours = True
            loaded_model = load_model(
                "savedmodels/best_model_autokeras_f1" + "_" + dataToPredict + "_" + timeframe)  # + str(forecast) + "_" + dataToPredict)#, custom_objects=ak.CUSTOM_OBJECTS)
            emit("modelexistornot", {"exist": "Predicting..."})
        else:
            detailledSecours = False
            loaded_model = load_model(
                "savedmodels/best_model_autokeras_f" + str(
                    forecast) + "_" + dataToPredict + "_" + timeframe)  # , custom_objects=ak.CUSTOM_OBJECTS)
            emit("modelexistornot", {"exist": "Predicting..."})

        evalmodel = evaluate_model(loaded_model, retrained, forecast, training_sample, allfeatures, y,
                                   detailled=detailled)
        prediction = predict_forecast(loaded_model, forecast, dataToPredict, scaler, allfeatures, detailled=detailled)
        return prediction.tolist()  # , evalmodel
    except Exception:
        if detailledSecours:
            print(f"Training f1_{dataToPredict}_{timeframe}")
            emit("modelexistornot", {"exist": "Training new model..."})
        else:
            print(f"Training f{forecast}_{dataToPredict}_{timeframe}")
            emit("modelexistornot", {"exist": "Training new model..."})
        train_model(forecast, training_sample, allfeatures, y, dataToPredict, timeframe, detailled=detailledSecours)
        predToMake = try_load_model(dataToPredict, retrained, forecast, training_sample,
                                    allfeatures, y, scaler, timeframe, detailled=detailledSecours)
        return predToMake


def retrain_model(retrained, forecast, training_sample, allfeatures, y, dataToPredict, scaler, timeframe):
    print(f"Training f1_{dataToPredict}_{timeframe}")
    emit("modelexistornot", {"exist": "Training new model..."})
    train_model(forecast, training_sample, allfeatures, y, dataToPredict, timeframe, detailled=True)
    print(f"Training f{forecast}_{dataToPredict}_{timeframe}")
    train_model(forecast, training_sample, allfeatures, y, dataToPredict, timeframe, detailled=False)
    predDetailled = try_load_model(dataToPredict, retrained, forecast, training_sample,
                                   allfeatures, y, scaler, timeframe, detailled=True)
    predStraight = try_load_model(dataToPredict, retrained, forecast, training_sample,
                                  allfeatures, y, scaler, timeframe, detailled=False)

    return predDetailled, predStraight


def train_model(forecast, trainingsample, allfeatures, y, dataToPredict, timeframe, detailled):
    stockpredictor = ak.StructuredDataRegressor(max_trials=3, overwrite=True, loss='mean_absolute_percentage_error')
    # input_node = ak.TimeseriesInput()
    # output_node = ak.DenseBlock(10)(output_node)
    # output_node = ak.RNNBlock(return_sequences=True, bidirectional=True, layer_type="ltsm")(input_node)
    # output_node = ak.DenseBlock(10)(output_node)
    # output_node = ak.RegressionHead(loss='mean_absolute_percentage_error')(output_node)
    # # stockpredictor = ak.AutoModel(
    # #     inputs=input_node, outputs=output_node, max_trials=3, overwrite=True
    # # )
    # stockpredictor = ak.AutoModel(inputs=input_node, outputs=output_node ,max_trials=3, overwrite=True, loss='mean_absolute_percentage_error')
    # # stockpredictor = ak.TimeseriesForecaster(inputs=input_node, outputs=output_node, max_trials=3, overwrite=True, loss='mean_absolute_percentage_error')
    # best loss functions: log_cosh > mean_absolute_percentage_error > mean_absolute_error
    epochs = 10
    if detailled:
        x_train = allfeatures[:trainingsample - 1]
        y_train = y[1:trainingsample]
        stockpredictor.fit(x_train, y_train, epochs=epochs)
        savedmodel = stockpredictor.export_model()
        try:
            savedmodel.save("savedmodels/best_model_autokeras_f1" + "_" + dataToPredict + "_" + timeframe,
                            save_format="tf")
        finally:
            savedmodel.save("savedmodels/best_model_autokeras_f1" + "_" + dataToPredict + "_" + timeframe + ".h5")
    else:
        x_train = allfeatures[:trainingsample - forecast]
        y_train = y[forecast:trainingsample]
        stockpredictor.fit(x_train, y_train, epochs=epochs)
        savedmodel = stockpredictor.export_model()
        try:
            savedmodel.save(
                "savedmodels/best_model_autokeras_f" + str(forecast) + "_" + dataToPredict + "_" + timeframe,
                save_format="tf")
        finally:
            savedmodel.save(
                "savedmodels/best_model_autokeras_f" + str(forecast) + "_" + dataToPredict + "_" + timeframe + ".h5")


def evaluate_model(model, retrained, forecast, trainingsample, allfeatures, y, detailled):
    if retrained:
        if detailled:
            x_test = allfeatures[trainingsample + 1:-1]
            y_test = y[trainingsample + 1 + 1:]
        else:
            x_test = allfeatures[trainingsample + 1:-forecast]
            y_test = y[trainingsample + 1 + forecast:]

        return model.evaluate(x_test, y_test)
    else:
        return model.evaluate(allfeatures[:-1], allfeatures[1:])


def predict_forecast(model, forecast, dataToPredict, scaler, allfeatures, detailled):
    if detailled:
        allfeatures = scaler.inverse_transform(allfeatures)
        allfeatures = np.reshape(allfeatures, (-1,))
        d = {dataToPredict: allfeatures}
        arr = pd.DataFrame(data=d)
        all_features = scaler.fit_transform(arr[[dataToPredict]].to_numpy())
        newest_feature = all_features[-1]
        pred = model.predict(newest_feature)
        df = {dataToPredict: pred[0][0]}
        arr = arr.append(df, ignore_index=True)

        for x in range(forecast - 1):
            topred = arr[[dataToPredict]].to_numpy()
            pred = scaler.fit_transform(topred)
            pred = model.predict(pred[-1])
            df = {dataToPredict: pred[0][0]}
            arr = arr.append(df, ignore_index=True)

        return arr[-forecast:].to_numpy()
    else:
        allfeatures = scaler.inverse_transform(allfeatures)
        allfeatures = np.reshape(allfeatures, (-1,))
        d = {dataToPredict: allfeatures}
        arr = pd.DataFrame(data=d)
        all_features = scaler.fit_transform(arr[[dataToPredict]].to_numpy())
        newest_feature = all_features[-1]
        pred = model.predict(newest_feature)

        return pred


def getOHLC(datasetname, retrained, forecast, timeframe):
    """

    :rtype: object
    """
    began = f"started at {time.strftime('%X')}"
    dates = getpredictions(datasetname, retrained, forecast, "open", timeframe=timeframe, getdates=True)
    emit("progressbar", {"progress": "5%"})
    opens = getpredictions(datasetname, retrained, forecast, "open", timeframe=timeframe)
    emit("progressbar", {"progress": "25%"})
    highs = getpredictions(datasetname, retrained, forecast, "high", timeframe=timeframe)
    emit("progressbar", {"progress": "50%"})
    lows = getpredictions(datasetname, retrained, forecast, "low", timeframe=timeframe)
    emit("progressbar", {"progress": "75%"})
    closes = getpredictions(datasetname, retrained, forecast, "close", timeframe=timeframe)
    emit("progressbar", {"progress": "100%"})
    print(began)
    print(f"finished at {time.strftime('%X')}")
    d = {"dates": dates, "open": opens, "high": highs, "low": lows, "close": closes}

    return d


# print("Opens: ", opens)
# print("Highs: ", highs)
# print("Lows: ", lows)
# print("Closes: ", closes)

# t = np.linspace(0, 2 * math.pi, len(y_test))

# print(getOHLC("Binance_BTCUSDT_1h.csv", False, 3, "hours"))
