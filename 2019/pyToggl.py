"""Connect to toggl account and fetch the year entries."""
import shelve
import os
import warnings
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
import requests
from decouple import config
from pandas import json_normalize


class YearEntries:
    """Get the time entries for the selected year.

    Starting on Jan 1st get entries in batchs of 1000 rows, concatenate them
    together in a Pandas dataFrame and clean it up.
    """

    token = config('API_TOKEN')
    workspace = config('WORKSPACE_ID')
    user_agent = config('USER_AGENT')
    headers = {'Content-type': 'application/json', }
    base_url = 'https://www.toggl.com/api/v8/'
    endpoints = dict(
        time_entries=base_url + 'time_entries',
        projects=base_url + 'workspaces/%s/projects' % workspace,
        tasks=base_url + 'tasks/',
    )

    def __init__(self, year):
        """Create the object."""
        if not isinstance(year, int):
            raise TypeError('Year should be an int')

        if year < 2019:
            raise ValueError('Yearshould be at least 2019')

        self.year = year

    def _get_1000_entries(self, startdate):
        """Get one thousand entries from the server."""
        params = {'workspace_id': self.workspace,
                  'user_agent': self.user_agent,
                  'start_date': startdate,
                  }
        url = self.endpoints['time_entries']
        resp = requests.get(
            url, headers=self.headers, params=params,
            auth=(self.token, 'api_token'))

        resp.raise_for_status()  # Raise an error in case

        return resp.json()

    def _get_task_name(self, task_id):
        """Return the name given the unique id."""
        if not isinstance(task_id, str):
            raise TypeError('TaskID should be a str.')

        url = self.endpoints['tasks'] + task_id
        resp = requests.get(url, headers=self.headers,
                            auth=(self.token, 'api_token'))
        resp.raise_for_status()
        return resp.json()['data']['name']

    def _cached_tasks(self, task_ids):
        """Cache and retrieve the dictionary of tasks."""
        if not isinstance(task_ids, np.ndarray):
            raise TypeError('TaskIDs should be a numpy array.')

        with shelve.open('t_cache') as d:
            for id in task_ids:
                id = str(int(id))
                if id not in d:
                    d[id] = self._get_task_name(id)

            # Get a read only copy
            t_dict = dict()
            for k, v in d.items():
                t_dict[k] = v

        return t_dict

    def rebuild_cache(self, df):
        """Rebuild the cache for tasks."""
        with shelve.open('t_cache') as cache:
            # first reset the dict
            for k in cache:
                del cache[k]

            # Now rebuild
            task_ids = pd.unique(df[df['tid'].notna()]['tid'])
            for id in task_ids:
                id = str(int(id))
                cache[id] = self._get_task_name(id)
        return df

    def clean_project(self, df):
        """Add project names to df."""
        # get projects df
        resp = requests.get(
            self.endpoints['projects'], headers=self.headers,
            auth=(self.token, 'api_token'))
        resp.raise_for_status()  # Raise an error in case
        projects = resp.json()
        pdf = json_normalize(projects).loc[:, ['id', 'name', ]]

        return df.merge(pdf, left_on='pid', right_on='id')

    def clean_task(self, df):
        """Replace task ids by task names in the df."""
        task_ids = pd.unique(df[df['tid'].notna()]['tid'])

        # Bulid a dictionary with ids and names
        t_dict = self._cached_tasks(task_ids)

        # Now replace each id by its name
        for key, value in t_dict.items():
            filter_in = (df['tid'] == float(key))
            df.loc[filter_in, 'tid'] = value

        # Finally rename the col
        return df.rename(columns={'tid': 'task'})

    def clean(self, df):
        """Clean and prepare the df to fiddle it."""
        # Set start and stop to DateTime object
        df['start'] = pd.DatetimeIndex(df['start'])
        df['stop'] = pd.DatetimeIndex(df['stop'])

        # add project names
        df = self.clean_project(df)

        # add task names
        df = self.clean_task(df)

        # Unpack tags into different columns
        tags = df['tags'].apply(pd.Series)
        tags = tags.rename(columns=lambda x: 'tag_' + str(x))
        df = df.join(tags)

        # Get rid of ongoing times
        df = df[df['duration'] >= 0]

        # remove duplicates
        df = df.drop_duplicates(subset='guid')

        # finally get rid of useless columns
        cols = ['at', 'duronly', 'uid', 'wid', 'tags', 'pid', 'id_y']
        df = df.drop(cols, axis=1)

        return df.sort_values(by=['start'])

    def fetch(self):
        """Bundle all the responses in a pd dataframe.

        This approach involves up-to-1000-row calls to the API util the full
        year's history is completed.
        """
        frames = list()

        # Build the initial start date
        start = datetime(self.year, 1, 1)
        start = start.isoformat(timespec='seconds') + '+01:00'

        loop = True
        while loop:
            data = self._get_1000_entries(start)
            tdf = json_normalize(data)
            frames.append(tdf)

            tdf['date'] = pd.DatetimeIndex(tdf['start']).date
            cur_date = tdf['date'].tail(1).values[0]

            # now check if we reached today
            if cur_date == date.today():
                loop = False
            else:
                start = datetime(
                    cur_date.year, cur_date.month, cur_date.day)
                start = start.isoformat(timespec='seconds') + '+01:00'

        df = pd.concat(frames, sort=True)

        df = self.clean(df)

        return df


class Utils:
    """Some useful tools to operate with pyToggl entries dataframe."""

    # Calculate elapsed days
    now = datetime.now()
    elapsed = (
        date.today() - date(now.date().year, 1, 1)).days  # Until yesterday
    elapsed += now.time().hour / 24  # The proportional part of today

    # Elapsed awake time estimated on 7h sleep basis
    awake = 17 * elapsed

    def __init__(self, df):
        """Init the object."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df parameter should be a DataFrame type')
        self.df = df

    @staticmethod
    def hr(seconds):
        """Display time in hours rounded to two decimals."""
        return round(seconds / 3600, 2)

    @staticmethod
    def goal(hours):
        """Calculate yearly goals progress in terms of elapsed days."""
        t = date.today()
        elapsed = (t - date(t.year - 1, 12, 31)).days
        return round(elapsed * hours / 365, 2)

    @staticmethod
    def perc(partial, total):
        """Calculate percentages between quantities."""
        return round(partial * 100 / total, 1)

    @staticmethod
    def daily(df):
        """Build a df with time duration per day adding zeroes in case."""
        # first create the date index as reference
        date_idx = pd.date_range(
            start=date(2019, 1, 1), end=date.today(), freq='D')
        base = pd.Series(np.zeros(len(date_idx)), date_idx).rename('zeroes')

        # now group math entries
        df = df.groupby('date').sum()

        # Join dfs
        df = pd.merge(
            base, df, how='left', left_index=True, right_index=True)

        # Get rid of columns & fill NaNs
        df = df.drop(columns=['zeroes', 'billable', 'id_x']).fillna(0)

        return df

    @staticmethod
    def custom_slice(df_len, step):
        """Filter in first, last, and the middle ones with step."""
        return [0] + [i for i in range(1, df_len - 2, step)] + [df_len - 1]

    def count_tag_cols(self):
        """Count how many tag cols there are in the df."""
        n = 0
        for i in range(20):
            col = 'tag_' + str(i)
            try:
                self.df[col]
            except KeyError:
                break
            else:
                n += 1
        return n

    def search_tags(self, tag):
        """Return a tag filtered dataframe."""
        frames = list()
        tag_cols = self.count_tag_cols()
        for col in range(tag_cols):
            col_name = 'tag_' + str(col)
            frames.append(self.df[self.df[col_name] == tag])
        return pd.concat(frames, sort=True).sort_values(by=['start'])

    def search_project(self, project):
        """Return the df with the time entries for a single project."""
        return self.df[self.df['name'] == project]

    def streak(self, project=None, tag=None):
        """Return a streak list for a certain project/tag."""
        if not project and not tag:
            raise TypeError('No project nor tag were provided')

        # get dates for project/tag
        if project:
            entries = self.search_project(project)
            name = project
        else:
            entries = self.search_tags(tag)
            name = tag
        if entries.empty:
            raise TypeError(
                'No entries returned, is \'%s\' written right?' % name)
        entries = entries.groupby('date').sum()
        dates_list = list(enumerate(entries.index[::-1]))

        max_delay = timedelta(hours=25)

        if date.today() - dates_list[0][1] > max_delay:
            streak_list = [0, ]
        else:
            streak_list = [1, ]

        for c, val in dates_list[:-1]:
            if val - dates_list[c + 1][1] > max_delay:
                streak_list.append(1)
            else:
                streak_list.append(streak_list[-1] + 1)

        return streak_list

    def cur_streak(self, streak_list):
        """Return the current streak status."""
        if streak_list[-1] == len(streak_list):
            return streak_list[-1]
        if streak_list[0] == 1:
            c_streak = 1
            for c, val in enumerate(streak_list[:-1]):
                if val >= streak_list[c + 1]:
                    c_streak = val
                    break
        elif streak_list[0] == 0:
            c_streak = 0
        else:
            raise ValueError('The first streak should be either 0 or 1')

        return c_streak

    def streak_is_done(self, project=None, tag=None):
        """Return if the streak today has been achieved."""
        if project:
            entries = self.search_project(project)
        elif tag:
            entries = self.search_tags(tag)
        else:
            raise TypeError('No project nor tag were provided')

        if entries.iloc[-1, 1] != date.today():
            return False
        else:
            return True

    def rates(self):
        """Prepare the dataframe to play with daily rates."""
        df = self.df[self.df['task'] == 'PnR']  # Filter in PnR task
        df = df[df['date'] != date.today()]  # Exclude today's entry (if any)

        # Exclude unrated entries
        valid_rates = [
            '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', ]
        df = df[df['tag_0'].isin(valid_rates)]

        # Convert rates to numbers
        df.loc[:, 'tag_0'] = pd.to_numeric(df['tag_0'])
        return df


class DataWarnings:
    """Display warnings about inconsistent data in df to fix in app."""

    def __init__(self, df):
        """Init the object."""
        self.df = df

    def duration_warning(self):
        """Show a warning with negative or zero durations."""
        if not self.df[self.df['duration'] <= 0].empty:
            warnings.warn('Wrong durations found df[\'duration\'<= 0]')
            return True
        else:
            return False

    def core_warning(self):
        """Warn for Core entries without Buildup."""
        df = self.df

        bu, co = 'BuildUp', 'Core'
        condition = df.columns.str.contains('tag')
        cols = df.columns[condition].to_list()

        condition = (df[cols] == co).any(axis=1)
        d0 = df[condition]

        condition = (d0[cols] == bu).any(axis=1)
        d0 = d0[~condition]
        if not d0.empty:
            warnings.warn(
                'Some entries found without BuildUp tag\n{}'.format(d0.date))
            return True

    def core_activities_warning(self, activities):
        df = self.df

        condition = df.columns.str.contains('tag')
        cols = df.columns[condition].to_list()

        condition = (df[cols].isin(activities)).any(axis=1)
        d0 = df[condition]

        condition = (d0[cols] == 'Core').any(axis=1)
        d0 = d0[~condition]
        if not d0.empty:
            warnings.warn(
                'Some entries found without Core tag\n{}'.format(d0.date))
            return True

    def rate_warning(self):
        """Find PnR entries without rate."""
        df = self.df[self.df['task'] == 'PnR']  # Filter in PnR
        df = df[df['date'] != date.today()]  # Exclude today's entry (if any)
        valid_rates = [
            '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', ]
        no_rate = df[~df['tag_0'].isin(valid_rates)]['date']
        if no_rate.any():
            warnings.warn('Some days found without rate\n{}'.format(no_rate))
            return True
        else:
            return False

    def show(self):
        """Launch the comprobations."""
        warn = self.duration_warning()
        warn = self.rate_warning()
        warn = self.core_warning()
        warn = self.core_activities_warning(['Python', 'Math', 'ML'])
        if not warn:
            print('No warnings, great!')


class CliExpress:
    """Get quick information of the year progress from the command line."""

    def __init__(self, reload=False):
        """Init the object.

        Reload option is used to refresh all the data from server.
        """
        self.utils = Utils
        self.reload = reload
        self.DF = self.display()

    def bu_vs_opk(self, df):
        """Return the hours for bu, core and billable activities."""
        utils = self.utils(df)
        output = dict()
        output['billable'] = df[df['billable'] == True]['duration'].sum()
        output['bu'] = utils.search_tags('BuildUp')['duration'].sum()
        output['core'] = utils.search_tags('Core')['duration'].sum()
        return output

    def cache(self):
        """Get previous years' data from a csv file

        It also reloads the csv file when self.reload is true or creates it
        when it does not exist.
        """
        f = 'older entries.csv'

        # fetch the data from file if it exists and don't want to refresh it.
        if os.path.isfile(f) and not self.reload:
            print('Using cache copy for older entries.')
            dtypes = {'task': 'category', }
            df = pd.read_csv(
                'older entries.csv', parse_dates=['start', 'stop'],
                dtype=dtypes, )
            # Convert the date col to datetimes and drop
            df['date'] = pd.DatetimeIndex(df.date).date
            df = df.drop(columns='Unnamed: 0')

        # Otherwise (re)create the file
        else:
            print('Reloading the data from server.')
            y = date.today().year
            old_dfs = [
                YearEntries(y).fetch() for y in range(2019, y)]
            df = pd.concat(old_dfs, sort=True)

            # Filter out current year entries
            df = df[~(pd.DatetimeIndex(df.start).year == y)]

            df.to_csv(f)

        return df

    def display(self):
        """Quick drop the info."""
        H = Utils.hr  # to convert into hours
        G = Utils.goal  # to calculate goals since jan 1st
        P = Utils.perc

        # Get data
        old = self.cache()
        curr = YearEntries(date.today().year).fetch()

        # Merge both sources of data to create a global dataframe
        DF = old.append(curr, sort=False)
        DF = DF.drop_duplicates('guid').reset_index(drop=True)

        # Show warnings (if any)
        DataWarnings(DF).show()

        # filter in this year data and export as attr
        df = DF[pd.DatetimeIndex(DF.start).year == 2020]
        self.df = df

        # Average sleep
        condition = df.name == 'ShiftSleep'
        avg = H(df[condition].duration.sum() / Utils.elapsed)
        print('\nAverage sleep per day: %sh' % avg)

        # Calculate awake time (only since 2020)
        condition = (df.name == 'ShiftSleep')
        aw = H(df[~condition].duration.sum())

        # And exclude those times from the df
        df = df[~condition]

        # Average time tracked and rates
        avg = H(df['duration'].sum() / Utils.elapsed)
        print('\nAverage time tracked per day: %sh' % avg)

        utils = self.utils(df)
        rates = utils.rates()['tag_0']
        ten, y = rates.tail(10).mean().round(2), rates.mean().round(2)
        print('Last 10 days/year rate average: {} / {}'.format(ten, y))

        print(50*'*')

        # Shared Time
        st = H(utils.search_tags('Iratxe')['duration'].sum())
        print('\nShared Time:    %sh (%s%%)\n' % (st, P(st, aw)))

        print('Billable vs BuildUp (% of awake time)\n' + 50*'*')

        # Billable vs R+D
        block = self.bu_vs_opk(df)
        opk, bu, core = H(block['billable']), H(block['bu']), H(block['core'])
        print('Billable time:  %sh (%s%%)' % (opk, P(opk, aw)))
        print('BuildUp time:   %sh (%sh) (%s%%)' % (
            bu, round(bu - opk, 2), P(bu, aw)))
        print('Core time:      %sh (%sh) (%s%%)' % (
            core, round(core - opk, 2), P(core, aw)))

        # Python
        py = H(utils.search_tags('Python').duration.sum())
        print('Python:         %sh (%s%% of core)' % (py, P(py, core)))

        # ML progress
        ml = H(utils.search_tags('ML').duration.sum())
        goal = G(450)
        diff = round(ml - goal, 2)
        print('ML time:        %sh; Goal: %sh; diff: %sh' % (ml, goal, diff))

        # Irontec progress
        irontec = H(utils.search_project('Warm Up').duration.sum())
        start = date(2020, 1, 1)
        date_range = (date(2020, 4, 30) - start).days
        goal = (150 / date_range) * (date.today() - start).days
        diff = round(irontec - goal, 2)
        print('Irontec time:   %sh; Goal: %sh; diff: %sh' % (
            irontec, goal, diff))

        print('\nStreaks  (max streak ever) *Uncompleted today\n' + 50*'*')

        """
        ## Streaks ##
        Streaks are calculated over total time tracked since 2019, so we have
        to instantiate utils with such df.
        """
        utils_all = Utils(DF)
        for tag in ('Core', 'Math', 'Python', 'ML'):
            dsp, ast = utils_all.streak(tag=tag), '*'
            if utils_all.streak_is_done(tag=tag):
                ast = ''
            print('%s: %s days (%s)%s' %
                  (tag, utils_all.cur_streak(dsp), max(dsp), ast))
        for p in ('typing course', 'Japanese'):
            dsp, ast = utils_all.streak(project=p), '*'
            if utils_all.streak_is_done(project=p):
                ast = ''
            print('%s: %s days (%s)%s' %
                  (p, utils_all.cur_streak(dsp), max(dsp), ast))

        return DF


if __name__ == '__main__':
    CliExpress()
