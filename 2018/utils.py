"""Some utitlities to explore the data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from calendar import month_abbr


def normalized(dataFrame):
    """Normalize the data got from Swipetimes.

    In order to show accumulated values in graphs, we should have a dataframe
    where each day has an entry for each project and the time tracked that day.
    This means that there will be days lacking times for some projects (and
    they should be 0).
    """
    # Get unique dates & project ids and the length column
    dates = dataFrame['started'].unique()  # len() = 365
    project_ids = dataFrame['project'].unique()  # len() = 22
    length = dataFrame['lenght'].values

    # Create the lists containing the columns
    nr_dates, nr_projects, nr_names, nr_tracked = (
        list(), list(), list(), list())

    # Join project id with name in a dict
    name_id_dict = dict()
    for p_id in project_ids:
        id_filter = (dataFrame['project'] == p_id)
        name_id_dict[p_id] = dataFrame['name'].values[id_filter][0]

    # Now, loop over to join them into rows (8030 entries)
    for date in dates:
        for project in project_ids:
            nr_dates.append(date)
            nr_projects.append(project)
            ev_day = (
                (dataFrame['started'] == date) &
                (dataFrame['project'] == project))
            tracked = np.sum(length[ev_day])
            nr_tracked.append(tracked)
            nr_names.append(name_id_dict[project])

    # Columns
    data = {'project': nr_projects, 'tracked': nr_tracked, 'name': nr_names}

    # Finally return the normalized dataFrame
    return pd.DataFrame(data, index=nr_dates)


def superDF(project_df, tag_df):
    """Get a normalized dataFrame with all the times in hours.

    We'll pack all the data from projects and tags in a dataFrame where there
    will be a column for each project and fo each tag and the index will be the
    dates. This means that there will be days lacking times for some projects
    and tags (and they should be 0).
    """
    # Get unique dates for index
    dates = project_df['started'].unique()  # len() = 365

    # Now, get project ids and tag names for columns names
    project_ids = project_df['name'].unique()  # len() = 22
    tag_names = tag_df['tag'].unique()  # len() = 19

    # finally, get the lenghts
    l_project = project_df['lenght'].values
    l_tag = tag_df['lenght'].values

    # get a list for each one of the projects
    data_p, data_t = defaultdict(list), defaultdict(list)
    for id in project_ids:
        data_p[id]
    for tag in tag_names:
        data_t[tag]

    # Build the project & tag lists
    for date in dates:
        for project in data_p.items():
            ev_day = (
                (project_df['started'] == date) &
                (project_df['name'] == project[0])
            )
            tracked = l_project[ev_day].sum()
            data_p[project[0]].append(tracked)
        for tag in data_t.items():
            ev_day = (
                (tag_df['started'] == date) &
                (tag_df['tag'] == tag[0])
            )
            tracked = l_tag[ev_day].sum()
            data_t[tag[0]].append(tracked)

    # Finally merge the two dictionaries
    data = {**data_p, **data_t}

    output = pd.DataFrame(data, index=dates)

    # retur timing in hours
    return output / 3600


def set_features(ax, title, x_axis, y_axis, axis='y'):
    """Set the main text in a graph in a single call."""
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.grid(axis=axis, color='darkslategray', linestyle='-.')
    return True


def groupedBarPlot(x, y_list, y_names, colors):
    """Create a multibar plot."""
    _, ax = plt.subplots(figsize=(15, 6))
    # Total width for all bars at one x location
    total_width = 0.8
    # width for each individual bar
    ind_width = total_width / len(y_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width/2), total_width/2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.bar(
            x + alteration[i], y_list[i], color=colors[i], 
            label=y_names[i], width=ind_width, )
    ax.legend(loc = 'upper right')
    return ax

def stacked_bar_plot(df, title, y_axis):
    """Create a stacked bar plot recursively"""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    cr = ('#7986cb',  # Indigo 300
          '#4db6ac',  # Teal 300
          '#cddc39',  # lime 500
          '#ffb74d',  # orange 300
          '#90a4ae',  # BlueGrey 300
          '#ba68c8',  # Purple 300
          '#e53935',  # red 600
         )

    
    df['height'] = 0  # height to be used as base for next column
    c_index = 0  # Color iterator
    
    # Now, iterate over columns
    for column in df.columns[:-1]:
        ax.bar(df.index, df[column],
               bottom=df['height'], label=column,
               color=cr[c_index]
              )
        df['height'] = df['height'] + df[column]
        c_index += 1
    
    # Finally, decorate it a bit
    set_features(ax, title, 'Months', y_axis,)
    plt.legend()
    plt.xticks(df.index, month_abbr[1:13])
    plt.show()
    
