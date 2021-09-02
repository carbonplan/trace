from datetime import datetime

import numpy as np
import scipy
import xarray as xr


def calc_rss(ytrue, ypred):
    return ((ypred - ytrue) ** 2).sum(dim='time').astype('float32')


def calc_fstat_pvalue(rss1, rss2, p1, p2, n, calc_p=False):
    """
    rss1 and p1 are from the restricted model
    rss2 and p2 are from the unrestricted model
    n is the number of samples
    """
    num = (rss1 - rss2) / (p2 - p1)
    denom = rss2 / (n - p2)
    f = (num / denom).astype('float32')
    del num, denom
    p = None
    if calc_p:
        p = 1 - scipy.stats.f.cdf(f, p2 - p1, n - p2)
    return f, p


def make_predictions_3D(x, pred, breakpoint, has_breakpoint, slope1, slope2, int1, int2):
    pred1 = (int1 + slope1 * x).transpose(*pred.dims).astype('float32')
    pred2 = (int2 + slope2 * x).transpose(*pred.dims).astype('float32')

    for i in range(1, len(x.time)):
        mask1 = has_breakpoint & (breakpoint == i) & (x < i)
        pred = xr.where(mask1, x=pred1, y=pred)

        mask2 = has_breakpoint & (breakpoint == i) & (x >= i)
        pred = xr.where(mask2, x=pred2, y=pred)
        del mask1, mask2
    del pred1, pred2

    return pred.astype('float32')


def linear_regression_3D(x, y, calc_p=True):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time.
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon).
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, and standard error on regression
    between the two datasets along their aligned time dimension.
    Lag values can be assigned to either of the data, with lagx shifting x, and lagy shifting y, with the specified lag amount.
    """
    # 1. Compute data length, mean and standard deviation along time axis for further use:
    n = len(x.time)
    y = y.astype('float32')
    ymean = y.mean(dim='time').astype('float32')

    if n >= 2:
        xmean = x.mean(dim='time')

        # 2. Compute covariance along time axis
        cov = (((x - xmean) * (y - ymean)).sum(dim='time') / n).astype('float32')

        # 5. Compute regression slope and intercept:
        slope = (cov / (x.std(dim='time') ** 2)).astype('float32')
        del cov
        intercept = (ymean - xmean * slope).astype('float32')

        # 6. Compute RSS
        pred = ((slope * x) + intercept).transpose(*y.dims).astype('float32')
        rss = calc_rss(y, pred).astype('float32')

        # 7. Compute F-stat and p value
        rss_null = calc_rss(y, ymean).astype('float32')
        del y
        fstat, pvalue = calc_fstat_pvalue(rss1=rss_null, rss2=rss, p1=1, p2=2, n=n, calc_p=calc_p)

        # polyfit & curvefit in xarray
        del xmean, ymean, pred, rss_null, fstat

    elif n == 1:
        zero_array = xr.DataArray(0, dims=ymean.dims, coords=ymean.coords)

        slope = zero_array
        intercept = ymean
        rss = zero_array
        pvalue = zero_array

        del ymean

    return slope.astype('float32'), intercept.astype('float32'), rss.astype('float32'), pvalue


def perform_change_detection(da):
    # 1. initialize parameter values
    print(f'1. {datetime.now()}')
    # this assumes that we're performing chow test for a time series with no additional independent variables
    # thus degree of freedom (k) = 2 (intercept, slope)
    k = 2
    # critical values were taken from Andrews (1993) for p (DoF) = 2
    # for a 7 point time series, our pi_0 is 1 / 7 = 0.14 (minimum 1 time point in each before/after group)
    # interpolating between pi_0 of 0.1 and 0.15 available on the table to find these values
    n = len(da.time)
    assert n == 7
    critical_value = 12.27  # 11.81 # 95% CI
    #     critical_value = 10.03  # 90% CI

    # 2. initialize x array as the independent variable (i.e. n timestep)
    print(f'2. {datetime.now()}')
    # subtracting 1 to be consistent with python index
    # TODO: have static numbers for
    x = xr.DataArray(np.arange(n), dims=['time'], coords=[da.coords['time']])
    da = da.astype('float32')

    # 3. fit one linear regression for entire time series
    print(f'3. {datetime.now()}')
    slope_total, int_total, rss_total, p_total = linear_regression_3D(x=x, y=da, calc_p=True)
    pred_total = (int_total + slope_total * x).transpose(*da.dims).astype('float32').compute()
    print(pred_total)
    del slope_total, int_total, p_total

    # 4. for each break point, fit 2 linear regression model and assess the fit, save the best fit
    print(f'4. {datetime.now()}')
    for i in range(1, n):
        print(f'4.{i} {datetime.now()}')
        slope1, int1, rss1, _ = linear_regression_3D(
            x=x.isel(time=slice(None, i)), y=da.isel(time=slice(None, i)), calc_p=False
        )
        slope2, int2, rss2, _ = linear_regression_3D(
            x=x.isel(time=slice(i, None)), y=da.isel(time=slice(i, None)), calc_p=False
        )

        # calculate f stat comparing model with break point (2 linear regressions) vs. without (1 linear regression)
        f_breakpoint, _ = calc_fstat_pvalue(
            rss1=rss_total, rss2=(rss1 + rss2), p1=k, p2=2 * k, n=n, calc_p=False
        )
        del rss1, rss2

        # if the current f stat is larger than the current max f for a pixel, save the current values
        if i == 1:
            max_f = f_breakpoint.astype('float32')
            breakpoint = xr.DataArray(
                i,
                dims=max_f.dims,
                coords=max_f.coords,
            ).astype('int8')
            output_slope1, output_slope2, output_int1, output_int2 = slope1, slope2, int1, int2
        else:
            mask = (f_breakpoint > max_f).compute()
            max_f = xr.where(mask, x=f_breakpoint, y=max_f)
            del f_breakpoint
            breakpoint = xr.where(mask, x=i, y=breakpoint)
            output_slope1 = xr.where(mask, x=slope1, y=output_slope1)
            output_slope2 = xr.where(mask, x=slope2, y=output_slope2)
            output_int1 = xr.where(mask, x=int1, y=output_int1)
            output_int2 = xr.where(mask, x=int2, y=output_int2)
            del mask

        del slope1, slope2, int1, int2

    del rss_total

    # 5. If the best fit from break point regression is better than the critical f value, make predictions based on that model
    print(f'5. {datetime.now()}')
    # else make prediction based on the 1 regression model
    has_breakpoint = max_f > critical_value
    del max_f
    pred = make_predictions_3D(
        x=x,
        pred=pred_total,
        breakpoint=breakpoint,
        has_breakpoint=has_breakpoint,
        slope1=output_slope1,
        slope2=output_slope2,
        int1=output_int1,
        int2=output_int2,
    ).transpose(*da.dims)
    print(pred)

    del (
        output_slope1,
        output_slope2,
        output_int1,
        output_int2,
    )

    # 6. If we think there is a break point, get p value for the 2 piece, otherwise save the p value for 1 linear regression
    print(f'6. {datetime.now()}')
    rss = calc_rss(da, pred)
    ymean = da.mean(dim='time')
    rss_null = calc_rss(da, ymean)
    del da
    _, p_total = calc_fstat_pvalue(rss1=rss_null, rss2=rss, p1=1, p2=k, n=n, calc_p=True)
    _, p_breakpoint = calc_fstat_pvalue(rss1=rss_null, rss2=rss, p1=1, p2=2 * k, n=n, calc_p=True)
    pvalue = xr.where(has_breakpoint, x=p_breakpoint, y=p_total)
    pvalue = pvalue.astype('float32')
    del rss, rss_null, p_breakpoint, p_total

    # 7. Update predictions based on p value
    print(f'7. {datetime.now()}')
    pred = xr.where(pvalue <= 0.05, x=pred, y=ymean)
    pred = pred.astype('float32')

    return pred, pvalue, breakpoint.where(has_breakpoint)
