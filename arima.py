import math
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta


class ArimaModel:
    def __init__(self, data, period):
        self.my_dat = data
        self.period = period
        self.my_res = None
        self.my_model = None
        self.dbData = None

    def checkData(self):
        max_day = self.my_dat.index.max()
        min_day = self.my_dat.index.min()
        if max_day - min_day <= timedelta(days=730):
            return "The cryptocurrency is relatively new or the data available is less than two years, \
            the model may not be reliable"
        else:
            return "The data is sufficient"

    def checkStationarity(self):
        result = adfuller(self.dbData)
        if result[1] >= 0.05:
            warning = "The P-value of the yield series is greater than 0.05, \
                it is considered non-stationary and the model is not reliable."
        else:
            warning = "This P-value is < 0.05. The Yield series is stationary"

        return warning, result[0], result[1]

    def createDataReturn(self):
        self.dbData = pd.DataFrame(np.log(self.my_dat['close'] / self.my_dat['close'].shift(1)))
        self.dbData = self.dbData.fillna(self.dbData.head().mean())
        return self.dbData

    def displaySummary(self):
        model = auto_arima(self.dbData, start_p=1, start_q=1,
                           max_p=10, max_q=10, m=1,
                           start_P=0, seasonal=False,
                           d=0, D=0, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False, max_order=10)

        self.my_model = SARIMAX(self.dbData, order=model.order)
        self.my_res = self.my_model.fit(disp=False)

        return self.my_res


    def predict(self, delta):
        period_lookup = {"1 DAY": 1, "1 WEEK": 7, "2 WEEKS": 14, "1 MONTH": 30}
        period = period_lookup[self.period]

        latest = self.my_dat.index.max() + timedelta(days=period)
        date_list = [latest + timedelta(days=x * period) for x in range(delta)]

        fc = self.my_res.get_prediction(start=int(self.my_model.nobs),
                                        end=self.my_model.nobs + delta - 1,
                                        full_reports=True)

        prediction = fc.predicted_mean
        prediction_ci = fc.conf_int()

        prediction = pd.DataFrame(prediction)
        prediction.index = date_list

        prediction_ci = pd.DataFrame(prediction_ci)
        prediction_ci.index = date_list

        prediction.columns = ['predicted_mean']
        lst_mean = self.actualPrice(list(prediction['predicted_mean']))
        lst_upper = self.actualPrice(list(prediction_ci['upper close']))
        lst_lower = self.actualPrice(list(prediction_ci['lower close']))

        date_list_predict = [self.my_dat.index.max() + timedelta(days=x * period) for x in range(delta + 1)]

        data_predict = pd.DataFrame({
            "Mean_Price": lst_mean,
            "Upper_Price": lst_upper,
            "Lower_Price": lst_lower
        }, index=date_list_predict)

        return data_predict


    def actualPrice(self, lst):
        # Get the last close price from the data
        last_price = list(self.my_dat['close'].iloc[[0]])
        expected_return = list(math.e ** self.dbData['close'].iloc[[0]])

        for i in lst:
            a = math.e ** i
            expected_return.append(a)
        for i in expected_return:
            x = last_price[-1] / i
            last_price.append(x)
        last_price.pop()
        return last_price
