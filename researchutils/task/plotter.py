import matplotlib.pyplot as plt


def _draw_task(id, start_time, end_time, color, task_name, line_width):
    plt.hlines(id, start_time, end_time, colors=color, lw=line_width)
    label_x = (start_time + end_time) / 2.0
    label_y = id
    plt.text(label_x, label_y, task_name, ha='center')


def draw_task(id, start_time, end_time, color, task_name, title, line_width=40, block=True):
    """
    Display timeline of given task as bar

    Parameters
    -------
    id : int
        Id of the task to draw. Correspond to the y-axis position
    start_time
        start time of given task
    end_time
        end time of given task
    color : array_like of colors
    task_name : str
        name of the task. will be displayed in the middle of bar
    title : str
        Title of the image to display
    line_width
        width of bar to draw
    block : bool, default True
        Block until the window closes
    """
    _draw_task(id, start_time, end_time, color, task_name, line_width)
    plt.title(title)
    plt.show(block=block)
