import matplotlib
import matplotlib.pyplot
import numpy as np
from researchutils.image import viewer

import pytest
from mock import patch


class TestViewer(object):
    def test_create_window(self):
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            image = np.ndarray(shape=(28, 28, 3))
            viewer._create_window(image=image)
            mock_imshow.assert_called_with(image)

            viewer._create_window(image=image, is_gray=True)
            mock_imshow.assert_called_with(image, cmap='gray')

    def test_show_image(self):
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            with patch('matplotlib.pyplot.title') as mock_title:
                with patch('matplotlib.pyplot.subplot') as mock_subplot:
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
                        title = 'test'
                        viewer.show_images(images=images, title=title)
                        assert mock_imshow.call_count == len(images)
                        assert mock_title.call_count == len(images)
                        mock_show.assert_called_once()

    def test_show_images_with_comparisons(self):
        with patch('matplotlib.pyplot.imshow') as mock_imshow:
            with patch('matplotlib.pyplot.title') as mock_title:
                with patch('matplotlib.pyplot.subplot') as mock_subplot:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        image1 = np.ndarray(shape=(28, 28, 3))
                        image2 = np.ndarray(shape=(28, 28, 3))
                        images = [image1, image2]
                        title = 'test'

                        image3 = np.ndarray(shape=(28, 28, 3))
                        image4 = np.ndarray(shape=(28, 28, 3))
                        comparisons = [image3, image4]
                        comparison_title = 'comparisons'

                        viewer.show_images(
                            images=images, title=title, comparisons=comparisons, comparison_title=comparison_title)
                        assert mock_imshow.call_count == len(images) + len(comparisons)
                        assert mock_title.call_count == len(images) + len(comparisons)
                        mock_show.assert_called_once()

    def test_animate(self):
        with patch('matplotlib.image.AxesImage') as mock_axes:
            with patch('matplotlib.pyplot.imshow', return_value=mock_axes) as mock_imshow:
                with patch('matplotlib.pyplot.figure') as mock_figure:
                    with patch('matplotlib.pyplot.show', return_value=mock_axes) as mock_show:
                        num_frames = 10
                        images = [np.ndarray(shape=(10, 10, 3))
                                  for i in range(num_frames)]
                        viewer.animate(images, auto_close=True)

                        assert mock_imshow.call_count == 1
                        assert mock_show.call_count == 1

    def test_animate_with_comparison(self):
        with patch('matplotlib.image.AxesImage') as mock_axes:
            with patch('matplotlib.pyplot.imshow', return_value=mock_axes) as mock_imshow:
                with patch('matplotlib.pyplot.figure') as mock_figure:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        with patch('matplotlib.pyplot.subplot') as mock_subplot:
                            num_frames = 10
                            images = [np.ndarray(shape=(10, 10, 3))
                                      for i in range(num_frames)]
                            viewer.animate(
                                images=images, comparisons=images, auto_close=True)

                            assert mock_imshow.call_count == 2
                            assert mock_subplot.call_count == 2
                            assert mock_show.call_count == 1

    def test_animate_save_gif(self):
        with patch('matplotlib.image.AxesImage') as mock_axes:
            with patch('matplotlib.pyplot.imshow', return_value=mock_axes) as mock_imshow:
                with patch('matplotlib.pyplot.figure') as mock_figure:
                    with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
                        with patch('matplotlib.pyplot.show') as mock_show:
                            num_frames = 10
                            images = [np.ndarray(shape=(10, 10, 3))
                                      for i in range(num_frames)]
                            viewer.animate(
                                images, auto_close=True, save_gif=True)

                            assert mock_save.call_count == 1
                            assert mock_imshow.call_count == 1
                            assert mock_show.call_count == 1


if __name__ == '__main__':
    pytest.main()
