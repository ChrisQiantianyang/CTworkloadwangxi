library(tidymodels)
library(modeltime)
library(modeltime.ensemble)
library(tidyverse)
library(timetk)
library(magrittr)
library(lubridate)
library(prophet)
library(knitr)
library(kableExtra)
library(here)
library(readxl)
library(caret)
library(fpp3)
datact = read_csv("D:/task/CTworkload/shiyan/datacount7pingsao.csv")
plot = datact %>%
  plot_time_series(ds,y)
datatrain = datact %>%
  filter(ds<as_date("2023-07-01"))
datatest = datact %>%
  filter(ds>=as_date("2023-07-01")&ds<as_date("2023-08-01"))
newyearsday = tibble(
  holiday = "newyearsday",
  ds = as_date(c('2022-12-31','2022-01-02','2023-01-02'
  )),
  lower_window = 0,
  upper_window = 1
)
springfestival = tibble(
  holiday = "springfestival",
  ds = as_date(c('2023-01-23','2023-01-24','2023-01-25','2023-01-26','2023-01-27',
                 '2022-01-31','2022-02-01','2022-02-02','2022-02-03','2022-02-04')),
  lower_window = 0,
  upper_window = 1
)
tombsweepingfestival = tibble(
  holiday = "tombsweepingfestival",
  ds = as_date(c('2023-04-05',
                 '2022-04-05',
                 '2022-04-04'
  )),
  lower_window = 0,
  upper_window = 1
)
labourday = tibble(
  holiday = "labourday",
  ds = as_date(c('2023-05-01','2023-05-02','2023-05-03',
                 '2022-05-04','2022-05-02','2022-05-03'
  )),
  lower_window = 0,
  upper_window = 1
)
dragonboatfestival = tibble(
  holiday = 'dragonboatfestival',
  ds = as_date(c('2023-06-22','2023-06-23',
                 '2022-06-03'
  )),
  lower_window = 0,
  upper_window = 1
)
midautumnnationalday = tibble(
  holiday = 'midautumnnationalday',
  ds = as_date(c('2021-09-19','2021-09-20','2021-09-21',
                 '2021-10-01','2021-10-02','2021-10-03','2021-10-04','2021-10-05','2021-10-06','2021-10-07',
                 '2022-09-12',
                 '2022-10-03','2022-10-04','2022-10-05','2022-10-06','2022-10-07'
  )),
  lower_window = 0,
  upper_window = 1
)
holidays <- bind_rows(newyearsday,springfestival,tombsweepingfestival,labourday,dragonboatfestival,midautumnnationalday)
holidays
change = c('2022-01-29','2022-01-30','2022-04-02','2022-04-24','2022-05-07',
           '2022-10-08','2022-10-09','2023-01-28','2023-01-29','2023-04-23',
           '2023-05-06',"2022-12-22","2023-01-23",'2021-09-18','2021-09-26',
           '2021-10-09'
)
changepoints = as_date(change)
model_spec = prophet_boost(seasonality_yearly = TRUE,
                           seasonality_weekly = TRUE,
                           seasonality_daily = TRUE) %>%
  set_engine(engine = "prophet_xgboost",holidays = holidays,changepoints = changepoints)
model_fit_boostedprophet = model_spec %>%
  fit(log(y) ~ ds + as.numeric(ds) + month(ds),data = datatrain)
model_fit_prophet = prophet_reg(seasonality_yearly = TRUE,
                                seasonality_weekly = TRUE,
                                seasonality_daily = TRUE) %>%
  set_engine(engine = "prophet",holidays = holidays,changepoints = changepoints) %>%
  fit(y ~ ds, data = datatrain)
model_fit_arima = arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(y ~ ds, data = datatrain)
model_fit_arima_boosted = arima_boost(
  min_n = 2,
  learn_rate = 0.015) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(y ~ ds + as.numeric(ds) + factor(month(ds), ordered = F),data = datatrain)
model_fit_ets = exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(y ~ ds, data = datatrain)
model_seasonal_reg = seasonal_reg(mode = "regression") %>%
  set_engine(engine = "tbats") %>%
  fit(y~ds,data = datatrain) 
models_tbl <- modeltime_table(
  model_fit_boostedprophet,
  model_fit_prophet,
  model_fit_arima,
  model_fit_arima_boosted,
  model_fit_ets,
  model_seasonal_reg
)
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = datatest)
calibration_tbl %>%
  modeltime_forecast(
    new_data    = datatest,
    actual_data = datact
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25
  )
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
  )
refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = datact)

refit_tbl %>%
  modeltime_forecast(actual_data = datact) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
  )
cty = calibration_tbl$.calibration_data[[5]]
cty2 = calibration_tbl$.calibration_data[[1]]
y_true <- exp(cty2$.actual)
y_pred <- exp(cty2$.prediction)
r_squared <- postResample(y_pred, y_true)
r_squared
r_squared2 <- postResample(cty$.prediction, cty$.actual)
r_squared2
cty1 = cty %>%
  mutate(absres = abs(.residuals))
gapall = sum(cty1$absres)
gapall
gap = gapall/31
gap
sumall = sum(datatest$y)
ratio = gapall/sumall
ratio
