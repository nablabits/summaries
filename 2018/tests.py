"""Test suite."""

from unittest import TestCase, main
import getData
import utils
from pandas.core.frame import DataFrame


class TestNewDataSet(TestCase):
    """Test the NewDataSet module."""

    def test_default_target_is_project(self):
        """Test default target."""
        data = getData.NewDataSet()
        self.assertEqual(data.target, 'project')

    def test_default_db(self):
        """Test default db."""
        data = getData.NewDataSet()
        self.assertEqual(data.db.db_url, 'sqlite:///db.sqlite')

    def test_get_returns_pandas_df(self):
        """Get should return a Pandas DataFrame."""
        data_project = getData.NewDataSet()
        data_tag = getData.NewDataSet(target='tag')
        self.assertIsInstance(data_project.get(), DataFrame)
        self.assertIsInstance(data_tag.get(), DataFrame)

    def test_invalid_target_raises_error(self):
        """Allowed targets are project and tag."""
        data = getData.NewDataSet(target='void')
        with self.assertRaises(ValueError):
            data.get()


class TestUtils(TestCase):
    """Test the utils module."""

    @classmethod
    def setUpClass(cls):
        """Run once the normalization due to its consuption."""
        project_time = getData.NewDataSet()
        pt = project_time.get()
        cls.normalized = utils.normalized(pt)

    def test_normalized_returns_a_pandas_df(self):
        """Normalized should return a pandas dataframe."""
        self.assertIsInstance(self.normalized, DataFrame)



if __name__ == '__main__':
    main()
