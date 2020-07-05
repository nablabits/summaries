"""Extract all the data from a sqlite db to use with pandas."""
import records


class NewDataSet(object):
    """Get a pandas DataFrame with al the data extracted from db."""

    def __init__(self, target='project', dbfile='db.sqlite'):
        """Ensure that there is a target and db file."""
        self.db = records.Database('sqlite:///' + dbfile)
        self.target = target

    def get(self):
        """Get all the data in a pandas dataframe."""
        if self.target == 'project':
            df = self.db.query_file('entries_project.sql')
            return df.export('df')
        elif self.target == 'tag':
            df = self.db.query_file('entries_tag.sql')
            return df.export('df')
        else:
            raise ValueError('Unexpected target:', self.target)
