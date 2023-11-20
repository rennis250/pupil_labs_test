from psychopy import visual, event
import os
import pandas as pd
import numpy as np
import robsblobs

import zmq
import msgpack as serializer
import time
import socket

ip_address = "127.0.0.1"
port = 50020

def check_capture_exists(ip_address, port):
    """check pupil capture instance exists"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if not sock.connect_ex((ip_address, port)):
            print("Found Pupil Capture")
        else:
            print("Cannot find Pupil Capture")
            sys.exit()


def setup_pupil_remote_connection(ip_address, port):
    """Creates a zmq-REQ socket and connects it to Pupil Capture or Service
    to send and receive notifications.

    We also set up a PUB socket to send the annotations. This is necessary to write
    messages to the IPC Backbone other than notifications

    See https://docs.pupil-labs.com/developer/core/network-api/ for details.
    """
    # zmq-REQ socket
    ctx = zmq.Context.instance()
    pupil_remote = ctx.socket(zmq.REQ)
    pupil_remote.connect(f"tcp://{ip_address}:{port}")

    # PUB socket
    pupil_remote.send_string("PUB_PORT")
    pub_port = pupil_remote.recv_string()
    pub_socket = zmq.Socket(ctx, zmq.PUB)
    pub_socket.connect("tcp://127.0.0.1:{}".format(pub_port))

    return pupil_remote, pub_socket


def request_pupil_time(pupil_remote):
    """Uses an existing Pupil Core software connection to request the remote time.
    Returns the current "pupil time" at the timepoint of reception.
    See https://docs.pupil-labs.com/core/terminology/#pupil-time for more information
    about "pupil time".
    """
    pupil_remote.send_string("t")
    pupil_time = pupil_remote.recv()
    return float(pupil_time)


def measure_clock_offset(pupil_remote, clock_function):
    """Calculates the offset between the Pupil Core software clock and a local clock.
    Requesting the remote pupil time takes time. This delay needs to be considered
    when calculating the clock offset. We measure the local time before (A) and
    after (B) the request and assume that the remote pupil time was measured at (A+B)/2,
    i.e. the midpoint between A and B.

    As a result, we have two measurements from two different clocks that were taken
    assumingly at the same point in time. The difference between them ("clock offset")
    allows us, given a new local clock measurement, to infer the corresponding time on
    the remote clock.
    """
    local_time_before = clock_function()
    pupil_time = request_pupil_time(pupil_remote)
    local_time_after = clock_function()

    local_time = (local_time_before + local_time_after) / 2.0
    clock_offset = pupil_time - local_time
    return clock_offset


def measure_clock_offset_stable(pupil_remote, clock_function, n_samples=10):
    """Returns the mean clock offset after multiple measurements to reduce the effect
    of varying network delay.

    Since the network connection to Pupil Capture/Service is not necessarily stable,
    one has to assume that the delays to send and receive commands are not symmetrical
    and might vary. To reduce the possible clock-offset estimation error, this function
    repeats the measurement multiple times and returns the mean clock offset.

    The variance of these measurements is expected to be higher for remote connections
    (two different computers) than for local connections (script and Core software
    running on the same computer). You can easily extend this function to perform
    further statistical analysis on your clock-offset measurements to examine the
    accuracy of the time sync.
    """
    assert n_samples > 0, "Requires at least one sample"
    offsets = [
        measure_clock_offset(pupil_remote, clock_function) for x in range(n_samples)
    ]
    return sum(offsets) / len(offsets)  # mean offset


def send_trigger(pub_socket, trigger):
    """Sends annotation via PUB port"""
    payload = serializer.dumps(trigger, use_bin_type=True)
    pub_socket.send_string(trigger["topic"], flags=zmq.SNDMORE)
    pub_socket.send(payload)


def new_trigger(label, duration, timestamp):
    """Creates a new trigger/annotation to send to Pupil Capture"""
    return {
        "topic": "annotation",
        "label": label,
        "timestamp": timestamp,
        "duration": duration,
    }


def notify(pupil_remote, notification):
    """Sends ``notification`` to Pupil Remote"""
    topic = "notify." + notification["subject"]
    payload = serializer.dumps(notification, use_bin_type=True)
    pupil_remote.send_string(topic, flags=zmq.SNDMORE)
    pupil_remote.send(payload)
    return pupil_remote.recv_string()


# initialize pupil connection
# 1. Setup network connection
check_capture_exists(ip_address, port)
pupil_remote, pub_socket = setup_pupil_remote_connection(ip_address, port)

# # 2. Setup local clock function
local_clock = time.perf_counter

# # 3. Measure clock offset accounting for network latency
stable_offset_mean = measure_clock_offset_stable(
    pupil_remote, clock_function=local_clock, n_samples=10
)

pupil_time_actual = request_pupil_time(pupil_remote)
local_time_actual = local_clock()
pupil_time_calculated_locally = local_time_actual + stable_offset_mean
print(f"Pupil time actual: {pupil_time_actual}")
print(f"Local time actual: {local_time_actual}")
print(f"Stable offset: {stable_offset_mean}")
print(f"Pupil time (calculated locally): {pupil_time_calculated_locally}")

# # 4. Prepare and send annotations
# # Start the annotations plugin
notify(
    pupil_remote,
    {"subject": "start_plugin", "name": "Annotation_Capture", "args": {}},
)

monGamma_sRGB = np.array([2.2, 2.2, 2.2])
monxyY_sRGB = np.array([
    [0.6400, 0.3300, 0.2126*100],
    [0.3000, 0.6000, 0.7152*100],
    [0.1500, 0.0600, 0.0722*100]])
mon_sRGB = robsblobs.monitor.Monitor("sRGB")
mon_sRGB.set_monGamma(monGamma_sRGB)
mon_sRGB.set_monxyY(monxyY_sRGB)

# do calibration before start of experiment
# notify(pupil_remote, {"subject": "calibration.should_stop"})
# time.sleep(1)
# notify(pupil_remote, {"subject": "calibration.should_start"})

win = visual.Window(fullscr=True, color=(0, 0, 0), units="pix")
# win = visual.Window([800, 800], fullscr=False, color=(0, 0, 0), units="pix")
win_size = win.size
# print(win_size, win_size[0]/win_size[1])
aspr_correction_x = win_size[1]/win_size[0]

instructions = visual.TextStim(win, text="Match the circle's color to the image and press space to continue.")
instructions.draw()
win.flip()
event.waitKeys(keyList=["space"])

# notify(pupil_remote, {"subject": "calibration.should_stop"})

win.mouseVisible = False

image_dir = 'images'
image_list = os.listdir(image_dir)

ld, rg, yv = 0.0, 0.0, 0.0  # Midpoint in DKL space

mouse = event.Mouse(win=win)

data = []

circle = visual.Circle(win, radius=100, fillColor=[0, 0, 0], lineColor='black', lineWidth=2.5, pos=(win_size[0]/4, 0), colorSpace='rgb1')

# start a recording for experiment
pupil_remote.send_string("R")
pupil_remote.recv_string()
time.sleep(1.0)  # sleep for a few seconds, can be less

label = "trial_annotation"
duration = 0.0

trialn = 0

# Set up a loop to present each image
for image_file in image_list:
    image_path = os.path.join(image_dir, image_file)
    image = visual.ImageStim(win, image=image_path, pos=(-400, 0), size=(600, 600))
    
    trial_data = {'image': image_file, 'trajectory': [], 'final_dkl': None}

    win.mouseVisible = False

    # brief pause before start of trial
    win.flip()
    time.sleep(1)

    # Send an annotation. Start of trial
    # We send a timestamp sampled from the local clock (e.g. that corresponds to a
    # trigger event, or a stimulus that was presented). The clock offset that we
    # measured in step 3 will be added to the timestamp to correctly align it with
    # Pupil Time. The annotation will be saved to annotation.pldata if a recording is
    # running. The Annotation_Player plugin will automatically retrieve, display and
    # export all recorded annotations.
    local_time = local_clock()
    minimal_trigger = new_trigger(label, duration, local_time + stable_offset_mean)
    minimal_trigger["trial_status"] = "start of " + str(trialn)
    send_trigger(pub_socket, minimal_trigger)

    # Event loop for each trial
    while True:
        image.draw()
        circle.draw()
        win.flip()

        buttons = mouse.getPressed()
        if buttons[0] == 1:
            ld -= 0.005
            if ld < -1.7:
                ld = -1.7
        elif buttons[2] == 1:
            ld += 0.005
            if ld > 1.4:
                ld = 1.4

        # Get current mouse position and normalize it
        mouse_x, mouse_y = mouse.getPos()
        rg = mouse_x / (win_size[0]/2)
        yv = mouse_y / (win_size[1]/2)

        rgb = robsblobs.dkl.dkl2rgb(mon_sRGB, np.array([ld, rg, yv]))
        rgb[rgb < 0] = 0
        rgb[rgb > 1] = 1
        circle.fillColor = np.sqrt(rgb)

        # Record trajectory
        trial_data['trajectory'].append((ld, rg, yv))

        keys = event.getKeys()
        if 'space' in keys:
            minimal_trigger["trial_status"] = "end of " + str(trialn)
            send_trigger(pub_socket, minimal_trigger)

            trial_data['final_dkl'] = (ld, rg, yv)
            data.append(trial_data)

            trialn += 1

            break


# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)
df['trajectory'] = df['trajectory'].apply(lambda x: str(x))  # Convert lists to strings for CSV compatibility

df.to_csv('experiment_data.csv', index=False)

win.close()

# stop recording, end of experiment
pupil_remote.send_string("r")
pupil_remote.recv_string()