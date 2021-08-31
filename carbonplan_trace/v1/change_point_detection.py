import scipy
import xarray as xr


def calc_rss(ytrue, ypred):
    return ((ypred - ytrue) ** 2).sum(dim='time')


def calc_fstat_pvalue(rss1, rss2, p1, p2, n, calc_p=False):
    """
    rss1 and p1 are from the restricted model
    rss2 and p2 are from the unrestricted model
    n is the number of samples
    """
    num = (rss1 - rss2) / (p2 - p1)
    denom = rss2 / (n - p2)
    f = num / denom
    p = None
    if calc_p:
        p = 1 - scipy.stats.f.cdf(f, p2 - p1, n - p2)
    return f, p


def make_predictions_3D(
    x, breakpoint, has_breakpoint, slope_total, slope1, slope2, int_total, int1, int2
):
    pred = (int_total + slope_total * x).transpose(*x.dims).astype('float32')

    for i in range(1, len(x.time)):
        mask1 = has_breakpoint & (breakpoint == i) & (x < i)
        pred1 = (int1 + slope1 * x).transpose(*x.dims).astype('float32')
        pred = xr.where(mask1, x=pred1, y=pred)

        mask2 = has_breakpoint & (breakpoint == i) & (x >= i)
        pred2 = (int2 + slope2 * x).transpose(*x.dims).astype('float32')
        pred = xr.where(mask2, x=pred2, y=pred)

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
    ymean = y.mean(dim='time')

    if n >= 2:
        xmean = x.mean(dim='time')
        xstd = x.std(dim='time')

        # 2. Compute covariance along time axis
        cov = ((x - xmean) * (y - ymean)).sum(dim='time') / n

        # 5. Compute regression slope and intercept:
        slope = cov / (xstd ** 2)
        intercept = ymean - xmean * slope

        # 6. Compute RSS
        pred = ((slope * x) + intercept).transpose(*y.dims)
        rss = calc_rss(y, pred)

        # 7. Compute F-stat and p value
        rss_null = calc_rss(y, ymean)
        fstat, pvalue = calc_fstat_pvalue(rss1=rss_null, rss2=rss, p1=1, p2=2, n=n, calc_p=calc_p)

        del xmean, ymean, xstd, cov, pred, rss_null, fstat

    elif n == 1:
        zero_array = xr.DataArray(0, dims=ymean.dims, coords=ymean.coords)

        slope = zero_array
        intercept = ymean
        rss = zero_array
        pvalue = zero_array

    return slope.astype('float32'), intercept.astype('float32'), rss.astype('float32'), pvalue


def perform_change_detection(da):
    # 1. initialize parameter values
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
    # subtracting 1 to be consistent with python index
    x = xr.DataArray(1, dims=da.dims, coords=da.coords).cumsum(dim='time').astype('int8') - int(1)

    # 3. fit one linear regression for entire time series
    slope_total, int_total, rss_total, p_total = linear_regression_3D(x=x, y=da, calc_p=True)

    # 4. for each break point, fit 2 linear regression model and assess the fit, save the best fit
    for i in range(1, n):
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
            mask = f_breakpoint > max_f
            max_f = xr.where(mask, x=f_breakpoint, y=max_f)
            breakpoint = xr.where(mask, x=i, y=breakpoint)
            output_slope1 = xr.where(mask, x=slope1, y=output_slope1)
            output_slope2 = xr.where(mask, x=slope2, y=output_slope2)
            output_int1 = xr.where(mask, x=int1, y=output_int1)
            output_int2 = xr.where(mask, x=int2, y=output_int2)

        del slope1, slope2, int1, int2, rss1, rss2, f_breakpoint

    # 5. If the best fit from break point regression is better than the critical f value, make predictions based on that model
    # else make prediction based on the 1 regression model
    has_breakpoint = (max_f > critical_value).compute()
    pred = make_predictions_3D(
        x=x,
        breakpoint=breakpoint,
        has_breakpoint=has_breakpoint,
        slope_total=slope_total,
        slope1=output_slope1,
        slope2=output_slope2,
        int_total=int_total,
        int1=output_int1,
        int2=output_int2,
    )

    del (
        slope_total,
        output_slope1,
        output_slope2,
        int_total,
        output_int1,
        output_int2,
        rss_total,
        max_f,
        mask,
    )

    # 6. If we think there is a break point, get p value for the 2 piece, otherwise save the p value for 1 linear regression
    rss = calc_rss(da, pred)
    rss_null = calc_rss(da, da.mean(dim='time'))
    _, p_breakpoint = calc_fstat_pvalue(rss1=rss_null, rss2=rss, p1=1, p2=4, n=n, calc_p=True)
    pvalue = xr.where(has_breakpoint, x=p_breakpoint, y=p_total).compute()
    pvalue = pvalue.astype('float32')
    del rss, rss_null, p_breakpoint, p_total

    # 7. Update predictions based on p value
    pred = xr.where(pvalue <= 0.05, x=pred, y=da.mean(dim='time')).compute()
    pred = pred.astype('float32')

    return pred, pvalue, breakpoint.where(has_breakpoint)
