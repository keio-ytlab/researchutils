import matplotlib
import matplotlib.pyplot
import numpy as np
from researchutils.task import plotter

import pytest
from mock import patch


class TestPlotter(object):
    def test_draw_task(self):
        with patch('matplotlib.pyplot.hlines') as mock_hlines:
            with patch('matplotlib.pyplot.title') as mock_title:
                with patch('matplotlib.pyplot.text') as mock_text:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        id = 1
                        start_time = 1
                        end_time = 2
                        color = 'black'
                        task_name = 'test'
                        title = 'title'
                        line_width = 100

                        plotter.draw_task(id, start_time, end_time,
                                          color, task_name, title, line_width=line_width)

                        mock_title.assert_called_once()
                        mock_hlines.assert_called_once_with(
                            id, start_time, end_time, colors=color, lw=line_width)
                        mock_show.assert_called_once()


if __name__ == '__main__':
    pytest.main()
