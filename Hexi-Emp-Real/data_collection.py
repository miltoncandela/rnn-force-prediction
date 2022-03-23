# General Imports
import os
import csv
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Imports for Multi-processing
from colorama import Fore, Style
from multiprocessing import Process, Value

# Imports from Empatica
import pylsl
import socket
from scipy.fft import rfft, rfftfreq
import matplotlib.animation as animation

# Imports for Hexiwear
# from pexpect.popen_spawn import PopenSpawn
import pexpect
import sys

# Imports for Intel Real Sense

# # CODE FOR Multi-Processing # #

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel processes.
seconds = Value("i", 0)


# # Define Parallel Processes # #
# This function is the count up timer.


def timer(second):
    # First we initialize a variable that will contain the moment the timer began.
    time_start = time.time()
    while True:
        # The .get_lock() function is necessary since it ensures they are synchronized between both functions,
        # since they both access to the same variables.
        with second.get_lock():
            # We now calculate the time elapsed between start and now (should be approx. 1 second).
            second.value = int(time.time() - time_start)
            if second.value == 15:
                return
            print(second.value, end="\r")
        # Once we stored all the info and make the calculations, we sleep the script for
        # one second. This is the magic of the script, it executes every  ~1 second.
        time.sleep(1)  # 0.996


# # CODE FOR EMPATICA # #


def empatica(second, folder):
    global BVP_array, Acc_array, GSR_array, Temp_array, IBI_array
    global BVP_tuple, ACC_tuple, GSR_tuple, Temp_tuple, IBI_tuple
    global Temporal_BVP_array, Temporal_GSR_array, Temporal_Temp_array, Temporal_IBI_array, BVP_Graph_value
    global counter
    global x_BVP_val, x_GSR_val, x_Temp_val, x_IBI_val
    global y_BVP_val, y_GSR_val, y_Temp_val, y_IBI_val

    # VARIABLES USED TO STORE & GRAPH DATA
    BVP_array, Acc_array, GSR_array, Temp_array, IBI_array = [], [], [], [], []
    BVP_tuple, ACC_tuple, GSR_tuple, Temp_tuple, IBI_tuple = (), (), (), (), ()
    Temporal_BVP_array, Temporal_GSR_array, Temporal_Temp_array, Temporal_IBI_array = [], [], [], []
    BVP_Graph_value = None
    counter = 0  # Used to pop values from arrays to perform a "moving" graph.
    x_BVP_val, x_GSR_val, x_Temp_val, x_IBI_val = [], [], [], []
    y_BVP_val, y_GSR_val, y_Temp_val, y_IBI_val = [], [], [], []

    # SELECT DATA TO STREAM
    acc = True  # 3-axis acceleration
    bvp = True  # Blood Volume Pulse
    gsr = True  # Galvanic Skin Response (Electrodermal Activity)
    tmp = True  # Temperature
    ibi = True

    serverAddress = '127.0.0.1'  # 'FW 2.1.0' #'127.0.0.1'
    serverPort = 28000  # 28000 #4911
    bufferSize = 4096
    # The wristband with SN A027D2 worked here with deviceID 8839CD
    deviceID = '834acd'  # '8839CD' #'1451CD' # 'A02088' #'A01FC2'

    def connect():
        global s
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)

        print("Connecting to server")
        s.connect((serverAddress, serverPort))
        print("Connected to server\n")

        print("Devices available:")
        s.send("device_list\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Connecting to device")
        s.send(("device_connect " + deviceID + "\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Pausing data receiving")
        s.send("pause ON\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    connect()

    time.sleep(1)

    def suscribe_to_data():
        if acc:
            print("Suscribing to ACC")
            s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if bvp:
            print("Suscribing to BVP")
            s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if gsr:
            print("Suscribing to GSR")
            s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if tmp:
            print("Suscribing to Temp")
            s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if ibi:
            print("Suscribing to Ibi")
            s.send(("device_subscribe " + 'ibi' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        s.send("pause OFF\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    suscribe_to_data()

    def prepare_LSL_streaming():
        print("Starting LSL streaming")
        if acc:
            infoACC = pylsl.StreamInfo('acc', 'ACC', 3, 32, 'int32', 'ACC-empatica_e4')
            global outletACC
            outletACC = pylsl.StreamOutlet(infoACC)
        if bvp:
            infoBVP = pylsl.StreamInfo('bvp', 'BVP', 1, 64, 'float32', 'BVP-empatica_e4')
            global outletBVP
            outletBVP = pylsl.StreamOutlet(infoBVP)
        if gsr:
            infoGSR = pylsl.StreamInfo('gsr', 'GSR', 1, 4, 'float32', 'GSR-empatica_e4')
            global outletGSR
            outletGSR = pylsl.StreamOutlet(infoGSR)
        if tmp:
            infoTemp = pylsl.StreamInfo('tmp', 'Temp', 1, 4, 'float32', 'Temp-empatica_e4')
            global outletTemp
            outletTemp = pylsl.StreamOutlet(infoTemp)
        if ibi:
            infoIbi = pylsl.StreamInfo('ibi', 'Ibi', 1, 2, 'float32', 'IBI-empatica_e4')
            global outletIbi
            outletIbi = pylsl.StreamOutlet(infoIbi)

    prepare_LSL_streaming()

    time.sleep(1)

    def reconnect():
        print("Reconnecting...")
        connect()
        suscribe_to_data()
        stream()

    def stream():
        try:
            print("Streaming...")
            try:
                with second.get_lock():
                    # When the seconds reach 312, we exit the functions.
                    if (second.value == 15):
                        plt.close()
                        return
                response = s.recv(bufferSize).decode("utf-8")
                # print(response)
                if "connection lost to device" in response:
                    print(response.decode("utf-8"))
                    reconnect()
                samples = response.split(
                    "\n")  # Variable "samples" contains all the information collected from the wristband.
                # print(samples)
                # We need to clean every temporal array before entering the for loop.
                global Temporal_BVP_array
                global Temporal_GSR_array
                global Temporal_Temp_array
                global Temporal_IBI_array
                global flag_Temp  # We only want one value of the Temperature to reduce the final file size.
                flag_Temp = 0
                for i in range(len(samples) - 1):
                    try:
                        stream_type = samples[i].split()[0]
                    except:
                        continue
                    # print(samples)
                    if (stream_type == "E4_Acc"):
                        global Acc_array
                        global ACC_tuple
                        timestamp = float(samples[i].split()[1].replace(',', '.'))
                        data = [int(samples[i].split()[2].replace(',', '.')),
                                int(samples[i].split()[3].replace(',', '.')),
                                int(samples[i].split()[4].replace(',', '.'))]
                        outletACC.push_sample(data, timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)#Added in 02/12/20 to show values
                        ACC_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        Acc_array.append(ACC_tuple)
                    if stream_type == "E4_Bvp":
                        global BVP_tuple
                        global BVP_array
                        timestamp = float(samples[i].split()[1].replace(',', '.'))
                        data = float(samples[i].split()[2].replace(',', '.'))
                        outletBVP.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_BVP_array.append(data)
                        BVP_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        BVP_array.append(BVP_tuple)
                    if stream_type == "E4_Gsr":
                        global GSR_array
                        global GSR_tuple
                        timestamp = float(samples[i].split()[1].replace(',', '.'))
                        data = float(samples[i].split()[2].replace(',', '.'))
                        outletGSR.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_GSR_array.append(data)
                        GSR_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        GSR_array.append(GSR_tuple)
                    if stream_type == "E4_Temperature":
                        global Temp_array
                        global Temp_tuple
                        timestamp = float(samples[i].split()[1].replace(',', '.'))
                        data = float(samples[i].split()[2].replace(',', '.'))
                        outletTemp.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_Temp_array.append(data)
                        if flag_Temp == 0:
                            Temp_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), Temporal_Temp_array[0])
                            Temp_array.append(Temp_tuple)
                            flag_Temp = 1
                    if stream_type == "E4_Ibi":
                        global IBI_array
                        global IBI_tuple
                        timestamp = float(samples[i].split()[1].replace(',', '.'))
                        data = float(samples[i].split()[2].replace(',', '.'))
                        outletIbi.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        # print(data)
                        Temporal_IBI_array.append(data)
                        IBI_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        IBI_array.append(IBI_tuple)
                # We get the mean of the temperature and append them to the final array.
                # Temp_tuple = (datetime.now().isoformat(), np.mean(Temporal_Temp_array))
                # Temp_array.append(Temp_tuple)

                # We pause the acquisition of signals for one second
                # time.sleep(3)
            except socket.timeout:
                print("Socket timeout")
                reconnect()
        except KeyboardInterrupt:
            """
            #Debugging print variables
            print(BVP_array)
            print("*********************************************")
            print()
            print(Acc_array)
            print("*********************************************")
            print()
            print(GSR_array)
            print("*********************************************")
            print()
            print(Temp_array)
            print()
            """
            # print("Disconnecting from device")
            # s.send("device_disconnect\r\n".encode())
            # s.close()

    # stream()

    # MATPLOTLIB'S FIGURE AND SUBPLOTS SETUP
    """
    Gridspec is a function that help's us organize the layout of the graphs,
    first we need to create a figure, then assign a gridspec to the figure.
    Finally create the subplots objects (ax's) assigning a format with gs (gridspec).
    """
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Temperature")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Electrodermal Activity")
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title("Blood Volume Pulse")
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_title("IBI")
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.set_title("Fast Fourier Transform")

    # Animation function: this function will update the graph in real time,
    # in order for it to work properly, new data must be collected inside this function.
    def animate(frame):
        global BVP_array
        global GSR_array
        global Temp_array
        global IBI_array
        global Temporal_BVP_array
        global Temporal_GSR_array
        global Temporal_Temp_array
        global Temporal_IBI_array
        global counter
        stream()  # This is the function that connects to the Empatica.

        # x_BVP_val = np.linspace(0,len(Temporal_BVP_array)-1,num= len(Temporal_BVP_array))
        # x_GSR_val = np.linspace(0,len(Temporal_GSR_array)-1,num= len(Temporal_GSR_array))
        # x_Temp_val = np.linspace(0,len(Temporal_Temp_array)-1,num= len(Temporal_Temp_array))
        # x_IBI_val = np.linspace(0,len(Temporal_IBI_array)-1,num= len(Temporal_IBI_array))

        x_BVP_val = np.arange(0.015625, ((len(Temporal_BVP_array)) * 0.015625) + 0.015625, 0.015625)
        x_GSR_val = np.arange(0.25, ((len(Temporal_GSR_array)) * 0.25) + 0.25, 0.25)
        x_Temp_val = np.linspace(0, len(Temporal_Temp_array) - 1, num=len(Temporal_Temp_array))
        x_IBI_val = np.linspace(0, len(Temporal_IBI_array) - 1, num=len(Temporal_IBI_array))

        X = rfft(Temporal_BVP_array)
        xf = rfftfreq(len(Temporal_BVP_array), 1 / 64)

        # GRAPHING ASSIGNMENT SECTION
        # First the previous data must be cleaned, then we plot the array with the updated info.
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax1.set_ylim(25, 40)  # We fixed the y-axis values to observe a better data representation.
        # ax2.set_ylim(0, 0.5)
        ax3.set_ylim(-150, 150)
        ax4.set_ylim(0, 1)
        ax1.set_title("Temperature")
        ax2.set_title("Electrodermal Activity")
        ax3.set_title("Blood Volume Pulse")
        ax4.set_title("IBI")
        ax5.set_title("Fast Fourier Transform")
        ax1.set_ylabel("Celsius (°C)")
        ax2.set_ylabel("Microsiemens (µS)")
        ax3.set_ylabel("Nano Watt")
        ax4.set_ylabel("Seconds (s)")
        ax5.set_ylabel("Magnitude")
        ax1.set_xlabel("Samples")
        ax2.set_xlabel("Seconds")
        ax3.set_xlabel("Seconds")
        ax4.set_xlabel("Samples")
        ax5.set_xlabel("Frequency (Hz)")

        if (counter >= 2400):
            ax1.plot(x_Temp_val, Temporal_Temp_array, color="#F1C40F")
            ax2.plot(x_GSR_val[-200:], Temporal_GSR_array[-200:], color="#16A085")
            ax3.plot(x_BVP_val[-2000:], Temporal_BVP_array[-2000:])
            ax4.plot(x_IBI_val, Temporal_IBI_array, color='#F2220C')
            ax5.plot(xf, np.abs(X))

        else:
            ax1.plot(x_Temp_val, Temporal_Temp_array, color="#F1C40F")
            ax2.plot(x_GSR_val, Temporal_GSR_array, color="#16A085")
            ax3.plot(x_BVP_val, Temporal_BVP_array)
            ax4.plot(x_IBI_val, Temporal_IBI_array, color='#F2220C')
            ax5.plot(xf, np.abs(X))

        counter += 60

        # Here es where the animation is executed. Try encaspsulation allows us

    # to stop the code anytime with Ctrl+C.
    try:
        anim = animation.FuncAnimation(fig, animate,
                                       frames=500,
                                       interval=1000)
        # Once the Animation Function is ran, plt.show() is necesarry, 
        # otherwise it won't show the image. Also, plt.show() will stop the execution of the code 
        # that is located after. So if we want to continue with the following code, we must close the 
        # tab generated by matplotlib.   
        plt.show()

        # The next lines allow us to create a CSV file with data retrieved from E4 wristband.
        # This code is repeated if a KeyboardInterrupt exception arises as a redundant case
        # for storing the data recorded.
        with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueBVP'])
            writer.writerows(BVP_array)

        with open("{}/Raw/fileACC.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueACC'])
            writer.writerows(Acc_array)

        with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueEDA'])
            writer.writerows(GSR_array)

        with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueTemp'])
            writer.writerows(Temp_array)

        with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueIBI'])
            writer.writerows(IBI_array)

            # Once we have the data stored locally on CSV files, we store it in json files to send them through a socket.
        csvFilePath = 'fileBVP.csv'
        # jsonPost(csvFilePath, 'valueBVP')
        csvFilePath = 'fileACC.csv'
        # jsonPost(csvFilePath, 'valueACC')
        csvFilePath = 'fileEDA.csv'
        # jsonPost(csvFilePath, 'valueEDA')
        csvFilePath = 'fileTemp.csv'
        # jsonPost(csvFilePath, 'valueTemp')
        csvFilePath = 'fileIBI.csv'
        # jsonPost(csvFilePath, 'valueIBI')

        # These next instructions should be executed only once, and exactly where we want the program to finish.
        # Otherwise, it may rise a Socket Error. These lines also written below in case of a KeyBoardInterrupt 
        # exception arising.
        global s
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()

    except KeyboardInterrupt:
        print('hola')
        # The next lines allow us to create a CSV file with data retrieved from E4 wristband.
        with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueBVP'])
            writer.writerows(BVP_array)

        with open("{}/Raw/fileACC.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueACC'])
            writer.writerows(Acc_array)

        with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueEDA'])
            writer.writerows(GSR_array)

        with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueTemp'])
            writer.writerows(Temp_array)

        with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline='') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueIBI'])
            writer.writerows(IBI_array)

            # Once we have the data stored locally on CSV files, we store it in json files to send them through a socket.
        csvFilePath = 'fileBVP.csv'
        # jsonPost(csvFilePath, 'valueBVP')
        csvFilePath = 'fileACC.csv'
        # jsonPost(csvFilePath, 'valueACC')
        csvFilePath = 'fileEDA.csv'
        # jsonPost(csvFilePath, 'valueEDA')
        csvFilePath = 'fileTemp.csv'
        # jsonPost(csvFilePath, 'valueTemp')
        csvFilePath = 'fileIBI.csv'
        # jsonPost(csvFilePath, 'valueIBI')

        # We close connections
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()


# # CODE FOR Hexiwear # #


def hexiwear(second, folder):
    # Using Hexiwear with Python
    # Script to get the device data and append it to a file
    # Usage
    # python GetData.py <device>
    # e.g. python GetData.py "00:29:40:08:00:01"

    # ---------------------------------------------------------------------
    # function to transform hex string like "0a cd" into signed integer
    # ---------------------------------------------------------------------
    def hexStrToInt(hexstr):
        val = int(hexstr[0:2], 16) + (int(hexstr[3:5], 16) << 8)
        if ((val & 0x8000) == 0x8000):  # treat signed 16bits
            val = -((val ^ 0xffff) + 1)
        return val

    # ---------------------------------------------------------------------

    parsed_to_json = []

    DEVICE = "00:22:50:04:00:0D"  # hexiwear ALAS-Tecnologico de Monterrey

    if len(sys.argv) == 2:
        DEVICE = str(sys.argv[1])

    # Run gatttool interactively.
    child = pexpect.spawn("gatttool -I")

    # Connect to the device.
    print("Connecting to:"),
    print(DEVICE)

    NOF_REMAINING_RETRY = 3

    while True:
        try:
            with second.get_lock():
                # We now calculate the time elapsed between start and now (should be approx. 1 second).
                if second.value == 15:
                    return
            child.sendline("connect {0}".format(DEVICE))
            child.expect("Connection successful", timeout=5)
        except pexpect.TIMEOUT:
            NOF_REMAINING_RETRY = NOF_REMAINING_RETRY - 1
            if (NOF_REMAINING_RETRY > 0):
                print("timeout, retry...")
                continue
            else:
                print("timeout, giving up.")
                break
        else:
            print("Connected!")
            break

    if NOF_REMAINING_RETRY > 0:
        try:
            while True:
                # The .get_lock() function is necessary since it ensures they are synchronized between both functions,
                # since they both access to the same variables.
                with second.get_lock():
                    # We now calculate the time elapsed between start and now (should be approx. 1 second).
                    if second.value == 15:
                        return
                time.sleep(1)
                unixTime = int(time.time())
                unixTime += 60 * 60  # GMT+1
                unixTime += 60 * 60  # added daylight saving time of one hour

                # [{‘tiempo’: 0, ‘speed: 12’}, {‘tiempo’: 1, ’speed’: 67}]
                # parsed_to_json = []

                # open file
                # file = open("data.csv", "a")
                # if (os.path.getsize("data.csv")==0):
                #  file.write("Device\ttime\tAppMode\tBattery\tAmbient\tTemperature\tHumidity\tPressure\tHeartRate\tSteps\tCalorie\tAccX\tAccY\tAccZ\tGyroX\tGyroY\tGyroZ\tMagX\tMagY\tMagZ\n")
                file = open("{}/Raw/dataHexi.csv".format(folder), "a")
                if (os.path.getsize("{}/Raw/dataHexi.csv".format(folder), ) == 0):
                    file.write("Time,Temperature,HeartRate\n")

                # file.write(DEVICE)
                # file.write('Time:')
                file.write(str(unixTime))  # Unix timestamp in seconds
                file.write(",")

                child.sendline("char-read-hnd 0x43")
                child.expect("Characteristic value/descriptor: ", timeout=5)
                child.expect("\r\n", timeout=5)
                print("Temperature:  "),
                print(child.before),
                print(float(hexStrToInt(child.before[0:5])) / 100)
                # file.write('Temperature:')
                file.write(str(float(hexStrToInt(child.before[0:5])) / 100))
                file.write(",")

                child.sendline("char-read-hnd 0x52")
                child.expect("Characteristic value/descriptor: ", timeout=5)
                child.expect("\r\n", timeout=5)
                print('HeartRate:'),
                print(child.before),
                print(str(int(child.before[0:2], 16)))
                # file.write('HeartRate:')
                file.write(str(int(child.before[0:2], 16)))
                file.write(",")

                file.write("\n")
                file.close()

                print("Datos de hexiwear registrados!")
                # parsed_to_json.append({ 'Time': str(unixTime) , 'Temp': str(float(hexStrToInt(child.before[0:5]))/100), 'HeartRate':str(int(child.before[0:2],16))})

                # sys.exit(0)
            # else:
            # print("FAILED!")
            # sys.exit(-1)
        except KeyboardInterrupt:  # Terminamos programa+Generamos JSON
            sys.exit(0)


# # CODE FOR RealSense # #

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.


if __name__ == '__main__':

    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    folder_name = 'S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder_name)

    for sub_folder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder_name, sub_folder))

    # # Start processes # #
    process1 = Process(target=timer, args=[seconds])
    p = Process(target=empatica, args=[seconds, folder_name])  # Descomentar para Empatica
    q = Process(target=hexiwear, args=[seconds, folder_name])
    process1.start()
    p.start()  # Descomentar para Empatica
    # q.start() # Descomentar para Hexiwear
    process1.join()
    p.join()  # Descomentar para Empatica
    # q.join() # Descomentar para Hexiwear

    print(Fore.GREEN + 'Test finished successfully' + Style.RESET_ALL)

# # # # # # # Sources # # # # # # # #
# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value
