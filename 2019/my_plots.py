"""Define some reusable custom plots."""

import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


def boxplot(x, y):
    """Create a basic seaborn style boxplot.

    Parameters:
        x, numpy ndarray
        y, numpy ndarray

    Returns: a bokeh.plotting.figure.Figure that can be edited afterwards
    (e.g.: p.title.text = 'Foo')
    """

    # First we should join both sources of data
    assert x.shape == y.shape
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    # Build the dataframe
    df = pd.DataFrame({'x': x, 'y': y, })
    outliers = df.copy()  # reserve for later

    # Get mean, Q1 & Q3
    a0 = [
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 75),
        np.mean, np.min, np.max]

    df = df.groupby('x').y.agg(a0)
    df['Q1'] = df.iloc[:, 0]
    df['Q3'] = df.iloc[:, 1]
    df = df.rename(columns={'amin': 'Q0', 'amax': 'Q4'})
    df = df.drop(['<lambda_0>', '<lambda_1>'], axis=1)
    iqr0 = df.Q3 - df.Q1

    # Come up with the max bounds for the box (helpers)
    lower = (df.Q1 - 1.5 * iqr0)  # Least lower bound for the box
    upper = (df.Q3 + 1.5 * iqr0)  # Max upper bound for the box
    df['lower'] = lower
    df['upper'] = upper

    # Assign lower bound to Q0 wherever Q0 value is below it
    below = df[df.Q0 < df.lower]
    df.loc[below.index, 'Q0'] = below.lower

    # Assign upper bound to Q4 wherever Q4 value is above it
    above = df[df.Q4 > df.upper]
    df.loc[above.index, 'Q4'] = above.upper

    # We don't need them anymore
    df = df.drop(columns=['lower', 'upper'])

    # Finally get outliers
    outliers = outliers.set_index('x')
    outliers = outliers.join(df.Q0).join(df.Q4)
    f = ((outliers.y < outliers.Q0) | (outliers.y > outliers.Q4))
    outliers = outliers[f].reset_index()

    # So x is a column
    df = df.reset_index()
    outliers = outliers.reset_index()

    ############
    # Plotting #
    s0, s1 = ColumnDataSource(df), ColumnDataSource(outliers)
    tools = 'hover,crosshair,pan,wheel_zoom,box_zoom,reset,tap'
    p = figure(title='Foo', tools=tools, )

    # Outliers
    size = max([df.x.max(), outliers.y.max()]) / 2
    p.circle('x', 'y', size=size, fill_alpha=.6, source=s1, name='Outliers')

    # IQR
    p.vbar(
        x='x', top='Q3', bottom='Q1', width=.8, source=s0,
        fill_alpha=.6, name='IQR')

    # Caps calculation
    lenght = ((df.x.max() - df.x.min()) / df.x.count()) * .8
    x0, x1 = df.x - lenght / 2, df.x + lenght / 2

    # Upper bound
    p.segment(x0='x', y0='Q3', x1='x', y1='Q4', source=s0)
    p.segment(x0=x0, y0=df.Q4, x1=x1, y1=df.Q4, name='top_cap')

    # mean
    p.segment(x0=x0, y0=df['mean'], x1=x1, y1=df['mean'])

    # Lower bound
    p.segment(x0='x', y0='Q1', x1='x', y1='Q0', source=s0)
    p.segment(x0=x0, y0=df.Q0, x1=x1, y1=df.Q0)

    return p
