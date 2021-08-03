# Author: Milton Candela (https://github.com/milkbacon)
# Date: August 2021

# The current code uses the generated CSV from "read_jsons.py" to make an animated plot of given key points on the XY
# axis. It has an option of using multiprocessing in order to visualize the difference between scaled vs non-scaled.

from time import sleep
import pandas as pd
import matplotlib.pyplot as plt


def plot_coordinates(file_name, resolution, refresh_rate=0.1):
    """
    This functions takes the name of the file as an input (Student, Taekwondo, Pers1), the original resolution of that
    video, and an optional refresh rate from which the animated plot would be refreshed. It is worth noting that
    matplotlib function is not being used, instead it creates a base plot on where it adds additional markers, the speed
    on which these markers are being added is controlled by the refresh_rate parameter.

    :param string file_name:
    :param tuple resolution:
    :param float refresh_rate:
    """

    # Reads the CSV based on the encoded name, defined on "read_jsons.py"
    scaled = ' (Scaled)' if len(file_name.split('_')) == 2 else ''
    resolution = TARGET_RES if scaled == ' (Scaled)' else resolution
    df = pd.read_csv('JSON_to_CSVs/XYCoordinates_' + file_name + '.csv', index_col=0)

    # Initializes the plot using the resolution and file specifications provided
    ph, = plt.plot(-1, -1, marker='o', color='red', linestyle='')
    ax = plt.gca()
    ax.set_xlim([0, resolution[0]])
    ax.set_ylim([0, resolution[1]])
    ax.set_xlabel('X coordinate (Pixels)')
    ax.set_ylabel('Y coordinate (Pixels)')
    ax.set_title(file_name.split('_')[0] + "'s 2D key points" + scaled)

    # The next for loop iterates all the rows on the extracted DataFrame, this rows represent each frame extracted by
    # the OpenPose software. A vector is created because it is also required to repeat this process for each joint
    # in the DataFrame, and so the final vector, which contains all the XY positions of all markers, is plotted.
    for i in range(df.shape[0]):
        coord_x_vector, coord_y_vector = [], []
        frame = df.iloc[i, :]
        # Additional modifications must be done on the Y axis, because the outputted JSON files from the OpenPose
        # software, are inverse on the Y axis. And so they must be flipped on this axis to see the plot as the video.
        for marker in df.columns:
            coord_x, coord_y, acc = frame[marker].strip("()\s").replace(' ', '').split(',')
            if (float(coord_x) != 0) and (float(coord_y) != 0):
                coord_x_vector.append(float(coord_x))
                coord_y_vector.append(resolution[1] - float(coord_y) - 1)
        ph.set_xdata(coord_x_vector)
        ph.set_ydata(coord_y_vector)
        plt.pause(refresh_rate)


TARGET_RES = (299, 299)
RESOLUTIONS = {'Student': (1920, 1080), 'Taekwondo': (1280, 720), 'Hexiwear': (640, 352), 'Rescaled': TARGET_RES}
plot_coordinates('Taekwondo_Esc', RESOLUTIONS['Taekwondo'])

from multiprocessing import Process


def multi_plotting(names, resolutions):
    """
    The following function uses multiprocessing to create two processes, each one representing a different animation
    of the key points extracted from the CSVs. It can be used to validate the amount of information loss when rescaling
    the original resolution to a different resolution (bigger or smaller), a timer of 2 is set to move the plot.

    :param list names: List of strings that correspond to the files that would be read in order to make the plots.
    :param list resolutions: List of tuples specifying the resolutions that is need to plot each file.
    """

    if __name__ == '__main__':
        proc1 = Process(target=plot_coordinates, args=(names[0], resolutions[0]))
        proc2 = Process(target=plot_coordinates, args=(names[1], resolutions[1]))
        proc1.start()
        sleep(2)
        proc2.start()
        proc1.join(), proc2.join()


# Uncomment the following line of code to plot two different CSV files
# multi_plotting(['Taekwondo','Taekwondo_Esc'], [RESOLUTIONS['Taekwondo'], TARGET_RES])
