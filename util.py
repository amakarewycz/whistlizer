import functools

from itertools import takewhile

import math
import functools
from typing import Tuple
import librosa
from parsing import parseMinSec, stripComments

# threshold = 0.6

# sample rate / samples / second
sr = 44100 / 2

# window frame size / seconds / frame , 0.7 is the typical length of a whistle blow
frame_size_seconds = 0.7

# frame length / samples
frame_length = frame_size_seconds * sr

# sliding window hop /  hops / frame
hop_in_window_divisions = 2

# hop length = (seconds / frame) / ( hop / frame ) * (samples / second) ) = samples / hop
hop_length = frame_size_seconds / hop_in_window_divisions * sr


frame_length_c = math.ceil(frame_length)
# frame_length as integer
frame_length = frame_length_c

# hop_length as integer
hop_length = math.floor(frame_length_c / hop_in_window_divisions)


def window_fft(y, n_fft=2048, hop_length=12000):
    """
    Returns an array of ffts of input signal based on hop_length

    :param y: input signal array of samples
    :param n_fft: number of frequency bins for fft
    :param hop_length: length
    :return:
    """
    return librosa.stft(y, n_fft=n_fft, center=False, hop_length=hop_length)


def find_bounds(n_fft=2048, sr=sr, low=2000, high=4000, ts=3):
    """
    Find the index bounds in the fft for the frequency band between low and high
    :param n_fft: number of bins in fft
    :param sr: sample rate / samples / second
    :param low: low frequency threshold
    :param high: high frequency threshold
    :param ts: number of windows, informational to size the number of input features
    :return: tuple (low_index, high_index)
    """
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_ = None
    high_ = None
    j = 0
    for i in freq:
        if low_ is None:
            if i > low:
                low_ = j
        if low_ and high_ is None:
            if i > high:
                high_ = j - 1
        j = j + 1

    print(f"low={low_} - hi={high_} x {ts} cross {(high_ - low_) * ts} fft bins {len(freq)}")
    return (low_, high_)


def find_bounds_bands(n_fft=2048, ts=3):
    return (find_bounds(n_fft, low=2000, high=4096, ts=ts),
            find_bounds(n_fft, high=8000, low=6000, ts=ts))


def convert_judgement_in_seconds_to_frames(hop_length, judgements=[]):  # frame_size,
    """
    convert judgements to array of frame indices given hop_length and sample rate

    :param hop_length: offset of sliding windows in samples
    :param judgements: array of times in seconds with positive judgements
    :return: array of frame indices with positive judgmeents
    """
    for i in judgements:
        #        r = i * sr % frame_size  # sec * samples / sec % ( samples / frame ) = frame
        a = i * sr / hop_length  # sec * samples / sec / ( samples / frame ) = frame
        a = math.floor(a)
        yield iter(range(a, a + 2))



def positive_frames_to_secs(predictions, threshold):
    """
    :param predictions: array of probabilties, the index is frame number.
    :param threshold: when probabily is greater than threshold, consider the prediction positive
    :return: array of times in seconds (rounded to 3 decimal places) where the prediction is greater than threshold
    """
    time = 0
    results = []
    last_time = -1
    for i in predictions:
        if i > threshold:
            seconds = math.floor(time / sr * 1000) / 1000
            results.append(seconds)
        time = time + hop_length
    return results


def secs_to_min_secmillis(seconds: float) -> str:
    """
    :param seconds:  seconds
    :return: seconds in [MM:]SS.Milliseconds time format
    """
    if seconds <= 60:
        res = f"{seconds:.5}"
    else:
        seconds = (seconds) % 60
        res = f"{math.floor(seconds / 60)}:{math.floor(seconds):02}.{math.floor(1000 * (seconds - math.floor(seconds))):03}"
    return res


def smooth(x, y):
    """
    Accumulate y into x only if the value stored in x[-1] (prior y) is not overlapping, ie. within the length of a frame.

    The input signal is chopped into overlapping windows.  We don't want positives in two adjacent frames if they overlap.
    Smooth consolidates positives that are overlapping to a single positive.

    :param x: accumulator storing y values that are in non-overlapping window frames
    :param y:
    :return:
    """
    if len(x) > 0:
        prior = x[-1]
        if y - prior < frame_length / sr:
            # print(f"Tossing {y} {secs_to_min_secmillis(y)} got prior {prior} {secs_to_min_secmillis(prior)} {y - prior} within frame_length {frame_length / sr}")
            pass
        else:
            x.append(y)
    else:
        x.append(y)
    return x


def write_times(results, name):
    """
    Writes results to file with name name.

    :param results: An array of times in Minutes:Seconds.Milliseconds
    :param name: output file name
    :return: None
    """
    with open(f"{name}.whistle", "a") as f:
        for i in results:
            f.write(i)
            f.write("\n")
            # print(i)
    f.close()


def write_positives_to_file(predictions, positive_threshold, filename):
    """
    Write positive predictions in [MM:]SS[.Milliseconds] format to filename

    :param predictions: array of probabiliites for each frame
    :param positive_threshold: threshold prediction is positive
    :param filename: output filename
    :return:
    """
    a = [secs_to_min_secmillis(i) for i in
         functools.reduce(smooth, positive_frames_to_secs(predictions, positive_threshold), [])]
    write_times(a, filename)
    return a


def generate_m3u(name, secs, extension="MOV", label=None):
    """
    :param name: filename w/o extension of video stream being classified
    :param secs: array of cue positions in [MM:]SS.Milliseconds format to save in play
        2:08.091
        2:11.241
        2:40.289
        2:58.838
        3:19.837

    :param extension: filename extension of video stream being classified
    :param label: array of labels to append to title of each element in secs.  By default, each element is labelled One, Two, Three increasing to Twenty Nine.  Useful to add additional context like FP TP FN to the cue position.
    :return: Context of m3u file
    """
    track = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
             "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty", "Twenty One",
             "Twenty Two", "Twenty Three", "Twenty Four", "Twenty Five", "Twenty Six", "Twenty Seven", "Twenty Eight",
             "Twenty Nine", "..."]
    bookmark_tmp = "{name=Omi,time=100}"
    a = """
#EXTM3U
#EXTINF:115,NAME.MOV
#EXTVLCOPT:bookmarks=BOOKMARKS
NAME.MOV
"""
    bookmarks = ""
    count = 0
    for i in secs:
        tmp = ("," if count > 0 else "") + str(bookmark_tmp)
        title = track[count] if count < len(track) - 1 else track[-1]

        if label is not None:
            title = title + " " + label[count]

        bookmarks = bookmarks + tmp.replace("Omi", title).replace("100", str(f'{i:0.3f}'))
        count = count + 1
        # print(bookmarks)
        # print(i)
    a = a.replace("BOOKMARKS", bookmarks)
    a = a.replace("NAME", name)
    a = a.replace("MOV", extension)

    return a


def write_m3u(name, contents):
    """
    Save context to file with name name

    :param name: output file name for playlist in m3u format https://en.wikipedia.org/wiki/M3U, compatible with VLC player https://www.videolan.org/vlc/
    :param contents: output
    :return: None
    """
    with open(f"{name}.m3u", "a") as f:
        f.write(contents)
    f.close()



def read(filename: str) -> [float]:
    """
    Read a_file, with lines in [MM:]SS[.Milliseconds] format, return an array of corresponding seconds for each line
    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        lines = [parseMinSec(stripComments("#")(line.rstrip())) for line in f]
        return lines


def join(ground_truth, predictions, threshold):
    """
    Returns an array of Tuples, ( time in seconds, TP | FP | FN ) a comparison of the predictions against ground_truth.
    TN True Negates are derived elsewhere based on the total # of predictions.  See docs for smooth_confusion_matrix_ below

    Say Threshold = 1/2 a window frame, in seconds = 1/2 * 0.7s = 0.35s

    GroundTruth         Predictions

        1                      3.2
        3.1                    6
        4                      7
        9

              Join

                 (1, "FN")
                 (3.1, "TP")  From Threshold 3.2 is within 0.35s of 3.1, count the prediction as TP.
                 (4, "FN")
                 (6, "FP")
                 (7, "FP")
                 (9, "FN")



    :param ground_truth: is an array of times in seconds for the precise instant when each whistle impulse starts
    :param predictions: is an array of times in seconds the model predicts a whistle impulse, these will be at frame boundary.
    :param threshold: time in seconds when comparing a prediction against ground_truth and counting it as a true positive.
    :return:
    """
    c_l = len(ground_truth)
    p_l = len(predictions)
    results = [(-99, "")]
    c_i = 0
    p_i = 0
    r_i = 0

    while True:
        left = ground_truth[c_i] if c_i < c_l else None
        right = predictions[p_i] if p_i < p_l else None
        last = results[-1]
        lastval = last[0]

        if left and right:
            diff_lr = (left - right)
            diff_l = (left - lastval)
            diff_r = (right - lastval)

            if threshold > diff_lr >= 0:
                # TP Match

                if threshold > diff_r or threshold > diff_l:
                    results[-1] = (lastval, "TP1")
                else:
                    results.append((left, "TP2"))

                c_i = c_i + 1
                p_i = p_i + 1

                continue

            elif diff_l < diff_r:
                c_i = c_i + 1
                if threshold < diff_l:
                    results.append((left, "FN3"))
                    continue
                else:
                    if "FP" in last[1]:
                        results[-1] = (lastval, "TP4")


            elif diff_l > diff_r:
                p_i = p_i + 1
                if threshold < diff_r:
                    results.append((right, "FP5"))
                    continue
                else:
                    if "FN" in last[1]:
                        results[-1] = (lastval, "TP6")
        elif right:
            diff_r = (right - lastval)
            p_i = p_i + 1
            if threshold < diff_r:
                results.append((right, "FP7"))

            continue
        elif left:
            diff_l = (left - lastval)
            c_i = c_i + 1
            if threshold < diff_l:
                results.append((left, "FN8"))
        else:
            break
    # Return all but the first, which was initialzed with -99
    return results[1:]


import numpy as np


def smooth_join(predictions, ground_truth="5u5a9273.groundtruth.whistle.txt", positive_threshold=.5, frame_threshold=3):
    """
    :param predictions: array of probabilities that a frame contains a whisle impulse
    :param ground_truth: text file containing times, one instant per line, in Minutes:Seconds.Milliseconds of the instant when whistle impulse starts,
    :param positive_threshold: a value from 0 - 1 to consider a prediction as positive.
    :param frame_threshold: relaxation threshold in units of frames used when comparing positive predictions against ground truth.  Ground truth is in milliseconds and predictions are factors of frame length, .
    :return: an array of Tuples, ( time in seconds, TP | FP | FN ) a comparison of the predictions against ground_truth.
    TN True Negates are derived elsewhere based on the total # of predictions.  See docs for smooth_confusion_matrix_ below

    GroundTruth       Predictions
                               3.2
        1                      3.55
        3.1                    6
        4                      7
        9
             Smooth

    GroundTruth       Predictions

        1                      3.2  *  3.55 is removed due to smoothing
        3.1                    6
        4                      7
        9

              Join

             (1, "FN")    A whistle at second 1 was missed by the model, count 1 as False Negative
             (3.1, "TP")  From frame_threshold 3.2 is within a 3 (default frame_threshold) frame lengths, count the prediction as True Positive.
             (4, "FN")
             (6, "FP")   The model predicted a whisle at second 6, count this prediction as False Positive
             (7, "FP")
             (9, "FN")

    """
    return join(np.array(read(ground_truth)),
                # see documentation for smooth function above
                np.array([i for i in functools.reduce(smooth,
                                                      # convert predictions to array of seconds, for what is positive according to positive_threshold
                                                      positive_frames_to_secs(predictions, positive_threshold), [])]),
                # convert frame_threshold units to seconds
                (frame_length / sr) * frame_threshold)

def smooth_join2(predictions, ground_truth, positive_threshold=.5, frame_threshold=3):
    return join(np.array(ground_truth),
         # see documentation for smooth function above
         np.array([i for i in functools.reduce(smooth,
                                               # convert predictions to array of seconds, for what is positive according to positive_threshold
                                               positive_frames_to_secs(predictions, positive_threshold), [])]),
         # convert frame_threshold units to seconds
         (frame_length / sr) * frame_threshold)


def smooth_confusion_matrix_(outcomes, number_of_predictions):
    """
    Wraps sklearn.metrics confusion_matrix to take outcomes as an input

    :param outcomes: array of Tuples, ( time in seconds, TP | FP | FN )
    :param number_of_predictions: the total number of predictions, including TN true negatives.
    :return: a matrix the count of true negatives is
C0,0, false negatives is C1,0, true positives is
C1,1 and false positives is C0,1
    """
    ground_truth_true = list(map(lambda x: x[0], filter(lambda x: "TP" in x[1] or "FN" in x[1], outcomes)))
    predicted_true = list(map(lambda x: x[0], filter(lambda x: "TP" in x[1] or "FP" in x[1], outcomes)))

    # resize to the total number of predictions
    gt2 = np.zeros(dtype=int, shape=(number_of_predictions))
    pred2 = np.zeros(dtype=int, shape=(number_of_predictions))

    # elements that  stay 0 are TN, TrueNegatives  since Total = sum(TN + TP + FP + FN)
    np.put(gt2, np.round(ground_truth_true).astype(int), v=1.)
    np.put(pred2, np.round(predicted_true).astype(int), v=1.)

    from sklearn.metrics import confusion_matrix
    print("""Thus in binary classification, the count of true negatives is 
C0,0, false negatives is C1,0, true positives is 
C1,1 and false positives is C0,1 """)

    return confusion_matrix(gt2, pred2)


def smooth_confusion_matrix(predictions, ground_truth, positive_threshold=.5, frame_threshold=3):
    """
    Convenience method for deriving confusion matrix from

    :param predictions: array of probabilities that a frame contains a whistle impulse
    :param ground_truth: text file containing times, one instant per line, in Minutes:Seconds.Milliseconds of the instant when whistle impulse starts,
    :param positive_threshold: a value from 0 - 1 to consider a prediction as positive.
    :param frame_threshold: relaxation threshold in units of frames used when comparing positive predictions against ground truth.  Ground truth is in milliseconds and predictions are factors of frame length, .

    :return: a matrix, the count of true negatives is
C0,0, false negatives is C1,0, true positives is
C1,1 and false positives is C0,1
    """
    return smooth_confusion_matrix_(smooth_join(predictions, ground_truth, positive_threshold, frame_threshold),
                                    len(predictions))

def smooth_confusion_matrix2(predictions, ground_truth, positive_threshold=.5, frame_threshold=3):
    return smooth_confusion_matrix_(smooth_join2(predictions, ground_truth, positive_threshold, frame_threshold),
                                    len(predictions))



def featurize(video):
    name = video.split(".")

    audio = f"{name[0]}.mp3"
    import moviepy.editor as mp

    b = mp.VideoFileClip(video)
    b.audio.write_audiofile(audio)

    import librosa

    y,sr = librosa.load(audio)

    from scipy import signal

    b, a = signal.iirfilter(17, [2000, 7900], rs=60,fs=sr,
                            btype='band', analog=False, ftype='cheby2')
    y_f =signal.lfilter(b,a,y,axis=-1)


    z=librosa.util.frame(y_f,
                         frame_length=frame_length_c,
                         hop_length=math.floor(frame_length_c / hop_in_window_divisions),
                         axis=0,
                         writeable=False,
                         subok=False)

    from itertools import islice
    import numpy as np

    nfft = 512
    hop_length = math.ceil(frame_length_c)


    def sss(ar, index, n_fft, positive_example=True):
        return window_fft(ar[index], n_fft=nfft, hop_length=hop_length)


    acc = None

    kk = sss(z, 0, nfft, True)
    ((low_1, high_1), (low_2, high_2)) = find_bounds_bands(n_fft=nfft, ts=np.shape(kk)[1])

    for i in range(len(z)):  # islice(indices,0,5,None):
        kk = sss(z, i, nfft, True)
        kk = np.concatenate(np.abs(np.array(kk)[np.s_[low_1:high_1]]))
        if acc is not None:
            acc = np.vstack((acc, kk))  # kk),axis=0)
        else:
            acc = kk

    kk=acc
    return kk


def test_join():
    print(join([], [1., 3., 6.], .7))  # == [(1.0, 'FP'), (3.0, 'FP'), (6.0, 'FP')])
    print(join([.25, 3.1, 3.5, 6.], [1, 2., 3., 9.],
               .7))  # == [(0.25, 'FN4'), (1, 'FP3'), (2.0, 'FP3'), (3.0, 'TP2'), (6.0, 'FN4'), (9.0, 'FP6')])
    print(join([2., 3., 9.], [1., 3., 6.],
               .7))  # == [(1.0, 'FP3'), (2.0, 'FN4'), (3.0, 'TP2'), (6.0, 'FP3'), (9.0, 'FN4')])
    print(join([1.3, 2.1, 3., 9.], [.5, 3., 6.],
               .7))  # == [(0.5, 'FP3'), (1.3, 'FN4'), (2.1, 'FN4'), (3.0, 'TP2'), (6.0, 'FP3'), (9.0, 'FN4')])
    print(join([2., 3., 9.], [], .7))  # == [(2.0, 'FN1'), (3.0, 'FN1'), (9.0, 'FN1')])

    c = [7.338, 21.562, 33.817, 45.569, 66.751, 81.811, 87.374,
         95.635, 106.372, 118.191, 137.015, 153.638, 159.304, 173.087,
         178.589, 192.039, 197.92, 200.838, 215.226, 229.4, 241.838,
         276.132, 277.553, 291.586, 298.919, 310.036, 344.001, 352.031,
         362.101, 384.55, 389.165, 403.645, 415.387, 419.643, 430.483,
         450.491, 463.659, 485.99, 490.938]
    p = [5.949, 7.699, 20.998, 22.048, 36.747, 56.696, 82.594,
         91.344, 98.343, 191.087, 204.736, 215.936, 227.135, 235.184,
         248.833, 289.431, 307.98, 323.029, 328.278, 337.378, 347.177,
         350.677, 362.226, 364.326, 368.526, 369.926, 375.175, 383.225,
         397.224, 408.423, 412.623, 417.872, 430.822, 450.77, 463.719,
         486.468]

    p = [5.949, 7.699, 20.998, 22.048, 36.747, 56.696, 82.594,
         91.344, 92.044, 98.343, 191.087, 204.736, 215.936, 227.135,
         235.184, 237.984, 248.833, 286.281, 287.331, 289.431, 307.98,
         323.029, 328.278, 337.378, 347.177, 350.677, 362.226, 364.326,
         367.826, 369.926, 375.175, 377.275, 383.225, 397.224, 408.423,
         412.623, 415.773, 417.872, 430.822, 450.77, 463.719, 486.468]

    sr = 44100 / 2
    frame_size_seconds = 0.7
    frame_length = frame_size_seconds * sr

    # print(join2(c, p,(frame_length/sr)*2))
    print(list(filter(lambda x: "TP" in x[1][1], filter(lambda x: True or x[0] < (frame_length / sr),
                                                        functools.reduce(lambda x, y: x + [(y[0] - x[-1][1][0], y)],
                                                                         join(c, p, 0.7), [(0, (0, "Start"))])))))

    d = [(0, (0, 'Start')), (5.949, (5.949, 'FP3')), (1.3890000000000002, (7.338, 'FN4')),
         (0.36099999999999977, (7.699, 'FP3')), (13.299000000000001, (20.998, 'FP3')),
         (0.5640000000000001, (21.562, 'FN4')), (0.4859999999999971, (22.048, 'FP3')),
         (11.769000000000002, (33.817, 'FN4')), (2.9299999999999997, (36.747, 'FP3')),
         (8.822000000000003, (45.569, 'FN4')),
         (11.126999999999995, (56.696, 'FP3')), (10.055000000000007, (66.751, 'FN4')),
         (15.060000000000002, (81.811, 'FN4')), (0.782999999999987, (82.594, 'FP3')),
         (4.780000000000001, (87.374, 'FN4')),
         (3.969999999999999, (91.344, 'FP3')), (4.291000000000011, (95.635, 'FN4')),
         (2.7079999999999984, (98.343, 'FP3')),
         (8.028999999999996, (106.372, 'FN4')), (11.819000000000003, (118.191, 'FN4')),
         (18.823999999999984, (137.015, 'FN4')), (16.62300000000002, (153.638, 'FN4')),
         (5.665999999999997, (159.304, 'FN4')), (13.782999999999987, (173.087, 'FN4')),
         (5.5020000000000095, (178.589, 'FN4')), (12.49799999999999, (191.087, 'FP3')),
         (0.9519999999999982, (192.039, 'FN4')), (5.881, (197.92, 'FN4')), (2.9180000000000064, (200.838, 'FN4')),
         (3.897999999999996, (204.736, 'FP3')), (10.490000000000009, (215.226, 'FN4')),
         (0.710000000000008, (215.936, 'FP3')), (11.198999999999984, (227.135, 'FP3')),
         (2.265000000000015, (229.4, 'FN4')),
         (5.783999999999992, (235.184, 'FP3')), (6.653999999999996, (241.838, 'FN4')),
         (6.9950000000000045, (248.833, 'FP3')), (27.299000000000007, (276.132, 'FN4')),
         (1.4209999999999923, (277.553, 'FN4')), (11.877999999999986, (289.431, 'FP3')),
         (2.1550000000000296, (291.586, 'FN4')), (7.33299999999997, (298.919, 'FN4')),
         (9.061000000000035, (307.98, 'FP3')),
         (2.055999999999983, (310.036, 'FN4')), (12.992999999999995, (323.029, 'FP3')),
         (5.249000000000024, (328.278, 'FP3')), (9.099999999999966, (337.378, 'FP3')),
         (6.6229999999999905, (344.001, 'FN4')), (3.1760000000000446, (347.177, 'FP3')), (3.5, (350.677, 'FP3')),
         (1.353999999999985, (352.031, 'FN4')), (10.069999999999993, (362.101, 'FN4')), (0.125, (362.226, 'FP3')),
         (2.1000000000000227, (364.326, 'FP3')), (4.199999999999989, (368.526, 'FP3')),
         (1.3999999999999773, (369.926, 'FP3')), (5.249000000000024, (375.175, 'FP3')),
         (8.050000000000011, (383.225, 'FP3')), (1.3249999999999886, (384.55, 'FN4')),
         (4.615000000000009, (389.165, 'FN4')), (8.058999999999969, (397.224, 'FP3')),
         (6.420999999999992, (403.645, 'FN4')), (4.77800000000002, (408.423, 'FP3')),
         (4.199999999999989, (412.623, 'FP3')),
         (2.76400000000001, (415.387, 'FN4')), (2.4850000000000136, (417.872, 'FP3')),
         (1.7709999999999582, (419.643, 'FN4')), (10.840000000000032, (430.483, 'FN4')),
         (0.33899999999999864, (430.822, 'FP3')), (19.668999999999983, (450.491, 'FN4')),
         (0.27899999999999636, (450.77, 'FP3')), (12.88900000000001, (463.659, 'FN4')),
         (0.060000000000002274, (463.719, 'FP3')), (22.271000000000015, (485.99, 'FN4')),
         (0.47800000000000864, (486.468, 'FP3')), (4.46999999999997, (490.938, 'FN4'))]
