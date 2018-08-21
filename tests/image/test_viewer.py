import matplotlib.pyplot
import numpy as np
from researchutils.image import viewer

import pytest
from mock import patch


class TestViewer(object):
    def test_create_window(self):
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            image = np.ndarray(shape=(28, 28, 3))
            viewer.create_window(image=image)
            mock_imshow.assert_called_with(image)

            viewer.create_window(image=image, is_gray=True)
            mock_imshow.assert_called_with(image, cmap='gray')

    def test_show_image(self):
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            with patch('matplotlib.pyplot.title') as mock_title:
                with patch('matplotlib.pyplot.show') as mock_show:
                    image = np.ndarray(shape=(28, 28, 3))
                    title = 'test'
                    viewer.show_image(image=image, title=title)
                    mock_imshow.assert_called_with(image)
                    mock_title.assert_called_with(title)
                    mock_show.assert_called_once()

    def test_show_images(self):
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            with patch('matplotlib.pyplot.title') as mock_title:
                with patch('matplotlib.pyplot.subplot') as mock_subplot:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        image1 = np.ndarray(shape=(28, 28, 3))
                        image2 = np.ndarray(shape=(28, 28, 3))
                        images = [image1, image2]
                        title1 = 'test1'
                        title2 = 'test2'
                        titles = [title1, title2]
                        viewer.show_images(images=images, titles=titles)
                        assert mock_imshow.call_count == len(images)
                        assert mock_title.call_count == len(titles)
                        mock_show.assert_called_once()

    def test_show_images_wrong_length(self):
        image1 = np.ndarray(shape=(28, 28, 3))
        image2 = np.ndarray(shape=(28, 28, 3))
        images = [image1, image2]
        title1 = 'test1'
        titles = [title1]
        with pytest.raises(ValueError):
            viewer.show_images(images=images, titles=titles)


if __name__ == '__main__':
    pytest.main()
