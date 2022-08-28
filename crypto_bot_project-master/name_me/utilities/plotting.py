###############################################################################
############################### IMPORTS #######################################
###############################################################################

#external imports
import pandas as pd
import numpy as np
np.warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from datetime import timedelta

###############################################################################

# internal imports

###############################################################################
###############################################################################

def plot(df, profile):

    fig = go.Figure()
    
    fig.update_layout(
        showlegend    = True,
        width         = 1900,
        height        = 600,
        margin        = dict(t=25),
        paper_bgcolor = '#1e1e1e',
        plot_bgcolor  = '#1e1e1e',
        font          = dict(size=13, color='#e1e1e1')
    )    
    
    fig.update_xaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.update_yaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index,
            open  = df['__OPEN'],
            high  = df['__HIGH'],
            low   = df['__LOW'],
            close = df['__CLOSE'],
            name  = f'{profile["market"]}:{profile["resolution"]}'
        )
    )
    
    fig.write_html('tmp.html', auto_open=True)

def plot_target(df, target, profile):
    
    labels = target.get_labels(df).shift(1)
    
    long_mask  = (labels == 1)
    short_mask = (labels == -1)
    none_mask  = (labels == 0)
    
    long_but_not_really  = long_mask & (df['__CLOSE'] < df['__OPEN'])
    short_but_not_really = short_mask & (df['__CLOSE'] > df['__OPEN'])

    fig = go.Figure()
    
    fig.update_layout(
        showlegend    = True,
        width         = 1900,
        height        = 600,
        margin        = dict(t=25),
        paper_bgcolor = '#1e1e1e',
        plot_bgcolor  = '#1e1e1e',
        font          = dict(size=13, color='#e1e1e1')
    )    
    
    fig.update_xaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.update_yaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[none_mask],
            open  = df['__OPEN'][none_mask],
            high  = df['__HIGH'][none_mask],
            low   = df['__LOW'][none_mask],
            close = df['__CLOSE'][none_mask],
            name  = f'{profile["market"]}:{profile["resolution"]}'
        )
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[long_mask],
            open  = df['__OPEN'][long_mask],
            high  = df['__HIGH'][long_mask],
            low   = df['__LOW'][long_mask],
            close = df['__CLOSE'][long_mask],
            name  = 'long',
            increasing_fillcolor = '#0f9af7',
            decreasing_fillcolor = '#0f9af7',
            increasing_line_color = '#0f9af7',
            decreasing_line_color = '#0f9af7'
        )
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[short_mask],
            open  = df['__OPEN'][short_mask],
            high  = df['__HIGH'][short_mask],
            low   = df['__LOW'][short_mask],
            close = df['__CLOSE'][short_mask],
            name  = 'short',
            increasing_fillcolor = '#852fad',
            decreasing_fillcolor = '#852fad',
            increasing_line_color = '#852fad',
            decreasing_line_color = '#852fad'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            mode   = 'markers'    ,
            x      = df.index[long_but_not_really],
            y      = (df['__CLOSE'] + df['__OPEN']).div(2)[long_but_not_really],
            marker = dict(color='#ff0000', size=7),
            name   = 'wrongly long'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode   = 'markers'    ,
            x      = df.index[short_but_not_really],
            y      = (df['__CLOSE'] + df['__OPEN']).div(2)[short_but_not_really],
            marker = dict(color='#ff8000', size=7),
            name   = 'wrongly short'
        )    
    )
    
    fig.write_html('tmp.html', auto_open=True)
    
def debug_plot(profile, df, labels, facs):
    
    short_mask = (labels == -1)
    none_mask  = (labels == 0)
    long_mask  = (labels == 1)
    
    wrongly_long  = long_mask  & (df['__CLOSE'] < df['__OPEN'])
    wrongly_short = short_mask & (df['__CLOSE'] > df['__OPEN'])
    

    fig = go.Figure()
    
    fig.update_layout(
        showlegend    = True,
        width         = 1900,
        height        = 600,
        margin        = dict(t=25),
        paper_bgcolor = '#1e1e1e',
        plot_bgcolor  = '#1e1e1e',
        font          = dict(size=13, color='#e1e1e1')
    )    
    
    fig.update_xaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.update_yaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[none_mask],
            open  = df['__OPEN' ][none_mask],
            high  = df['__HIGH' ][none_mask],
            low   = df['__LOW'  ][none_mask],
            close = df['__CLOSE'][none_mask],
            name  = f'{profile["market"]}:{profile["resolution"]}'
        )
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[long_mask],
            open  = df['__OPEN' ][long_mask],
            high  = df['__HIGH' ][long_mask],
            low   = df['__LOW'  ][long_mask],
            close = df['__CLOSE'][long_mask],
            name  = 'long',
            increasing_fillcolor  = '#0f9af7',
            decreasing_fillcolor  = '#0f9af7',
            increasing_line_color = '#0f9af7',
            decreasing_line_color = '#0f9af7'
        )
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[short_mask],
            open  = df['__OPEN' ][short_mask],
            high  = df['__HIGH' ][short_mask],
            low   = df['__LOW'  ][short_mask],
            close = df['__CLOSE'][short_mask],
            name  = 'short',
            increasing_fillcolor  = '#852fad',
            decreasing_fillcolor  = '#852fad',
            increasing_line_color = '#852fad',
            decreasing_line_color = '#852fad'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            mode   = 'markers'    ,
            x      = df.index[wrongly_long],
            y      = (0.985 * df['__CLOSE_MA2'])[wrongly_long],
            marker = dict(color='#ff0000', size=7),
            name   = 'wrongly long'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode   = 'markers'    ,
            x      = df.index[wrongly_short],
            y      = (0.985 * df['__CLOSE_MA2'])[wrongly_short],
            marker = dict(color='#ff8000', size=7),
            name   = 'wrongly short'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[facs == 1],
            y             = 1.03 * np.max(df['__CLOSE']) * np.ones(np.shape(facs == 1)),
            marker_symbol = "diamond",
            marker        = dict(color='#0000FF', size=7),
            name          = 'neutral factor'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[facs < 1],
            y             = 1.03 * np.max(df['__CLOSE']) * np.ones(np.shape(facs < 1)),
            marker_symbol = "diamond",
            marker        = dict(color='#FF0000', size=7),
            name          = 'neg factor'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[facs > 1],
            y             = 1.035 * np.max(df['__CLOSE']) * np.ones(np.shape(facs > 1)),
            marker_symbol = "diamond",
            marker        = dict(color='#00FF00', size=7),
            name          = 'pos factor'
        )    
    )
    
    fig.write_html('tmp.html', auto_open=True)
    
def nn_debug_plot(
        filename : str,
        profile  : dict,
        df       : pd.DataFrame,
        labels   : pd.Series,
        preds    : pd.Series,
        stays    : pd.Series,
        switches : pd.Series,
        facs     : pd.Series
    ):
    
    labels = labels.shift(1)
    facs   = facs.shift(1)
    
    short_mask = (labels == -1)
    none_mask  = (labels ==  0)
    long_mask  = (labels ==  1)
    
    switches1 = switches == 1
    switches2 = switches == 2
    
    stays = stays.astype(bool)
    
    long_but_not_really  = long_mask & (df['__CLOSE'] < df['__OPEN'])
    short_but_not_really = short_mask & (df['__CLOSE'] > df['__OPEN'])
    
    # create figure
    fig = go.Figure()
    
    # update layout and style
    fig.update_layout(
        showlegend    = True,
        width         = 1900,
        height        = 600,
        margin        = dict(t=25),
        paper_bgcolor = '#1e1e1e',
        plot_bgcolor  = '#1e1e1e',
        font          = dict(size=13, color='#e1e1e1')
    )    
    
    fig.update_xaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    fig.update_yaxes(
        gridcolor = '#1f292f',
        showgrid  = True    
    )
    
    # plot shorts, nones and longs
    fig.add_trace(
        go.Candlestick(
            x     = df.index[short_mask],
            open  = df['__OPEN' ][short_mask],
            high  = df['__HIGH' ][short_mask],
            low   = df['__LOW'  ][short_mask],
            close = df['__CLOSE'][short_mask],
            name  = 'short',
            increasing_fillcolor  = '#852fad',
            decreasing_fillcolor  = '#852fad',
            increasing_line_color = '#852fad',
            decreasing_line_color = '#852fad'
        )
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[none_mask],
            open  = df['__OPEN' ][none_mask],
            high  = df['__HIGH' ][none_mask],
            low   = df['__LOW'  ][none_mask],
            close = df['__CLOSE'][none_mask],
            name  = f'{profile["market"]}:{profile["resolution"]}'
        )
    )
    
    fig.add_trace(
        go.Candlestick(
            x     = df.index[long_mask],
            open  = df['__OPEN' ][long_mask],
            high  = df['__HIGH' ][long_mask],
            low   = df['__LOW'  ][long_mask],
            close = df['__CLOSE'][long_mask],
            name  = 'long',
            increasing_fillcolor  = '#0f9af7',
            decreasing_fillcolor  = '#0f9af7',
            increasing_line_color = '#0f9af7',
            decreasing_line_color = '#0f9af7'
        )
    )
    
    # indicate misclassified candles
    fig.add_trace(
        go.Scatter(
            mode   = 'markers'    ,
            x      = df.index[long_but_not_really],
            y      = (0.985 * df['__CLOSE_MA2'])[long_but_not_really],
            marker = dict(color='#ff0000', size=7),
            name   = 'wrongly long'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode   = 'markers'    ,
            x      = df.index[short_but_not_really],
            y      = (0.985 * df['__CLOSE_MA2'])[short_but_not_really],
            marker = dict(color='#ff8000', size=7),
            name   = 'wrongly short'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers'    ,
            x             = df.index[sw_1] + timedelta(seconds=profile['resolution']/2),
            y             = 1.025 * df['__CLOSE_MA2'][sw_1],
            marker_symbol = "triangle-down",
            marker        = dict(color='#FFFF00', size=7),
            name          = 'single switch'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers'    ,
            x             = df.index[sw_2] + timedelta(seconds=profile['resolution']/2),
            y             = 1.025 * df['__CLOSE_MA2'][sw_2],
            marker_symbol = "triangle-down",
            marker        = dict(color='#11CC11', size=7),
            name          = 'double switch'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[sts] + timedelta(seconds=profile['resolution']/2),
            y             = 1.025 * df['__CLOSE_MA2'][sts],
            marker_symbol = "hexagon2",
            marker        = dict(color='#00FFFF', size=7),
            name          = 'stays'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[facs == 1] - timedelta(seconds=profile['resolution']/2),
            y             = 1.03 * np.max(df['__CLOSE']) * np.ones(np.shape(facs == 1)),
            marker_symbol = "diamond",
            marker        = dict(color='#0000FF', size=7),
            name          = 'zero factor'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[facs < 1] - timedelta(seconds=profile['resolution']/2),
            y             = 1.03 * np.max(df['__CLOSE']) * np.ones(np.shape(facs < 1)),
            marker_symbol = "diamond",
            marker        = dict(color='#FF0000', size=7),
            name          = 'neg factor'
        )    
    )
    
    fig.add_trace(
        go.Scatter(
            mode          = 'markers',
            x             = df.index[facs > 1] - timedelta(seconds=profile['resolution']/2),
            y             = 1.035 * np.max(df['__CLOSE']) * np.ones(np.shape(facs > 1)),
            marker_symbol = "diamond",
            marker        = dict(color='#00FF00', size=7),
            name          = 'pos factor'
        )    
    )
    
    fig.write_html(filename, auto_open=True)
