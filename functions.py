#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import signal


def construct_stimulus(stimulus, word_pres_dur=20, within_break=5, between_break=100):
    """
    Creates a dataframe, each row of which contains information about
    an individual frame during stimulus presentation.

    Parameters
    ----------
    sentences: list of strings
        The sentences to be presented.
    word_pres_dur: int, optional
        Specifies the duration of presentation for each word in frames. Default is 20.
    within_break: int, optional
        Specifies the duration of within-sentence breaks between words in frames. Default is 5.
    between_break: int, optional
        Specifies the duration of breaks between sentences in frames. Default is 100.

    Returns
    ----------
    out: pandas.core.frame.DataFrame
        Dataframe containing the following columns:
        - word: str, presented word
        - sentence: int, sentence ID
        - change: bool, indicates the appearance of a new word
        - time: None, column to store exact presentation times during stimulus presentation
    """

    shuffled_indices = np.random.choice(
        np.arange(len(stimulus)), size=len(stimulus), replace=False
    )

    return (
        pd.concat(
            [
                pd.concat(
                    [
                        pd.DataFrame([word] * word_pres_dur, columns=["word"]).append(
                            pd.DataFrame([{"word": ""} for k in range(within_break)])
                        )
                        for word in stimulus[i]["text"].split(" ")
                    ]
                )
                .append(pd.DataFrame([{"word": ""} for k in range(between_break)]))
                .assign(sentence=stimulus[i]["ID"])
                for i in shuffled_indices
            ]
        )
        .reset_index(drop=True)
        .assign(change=lambda df: df["word"].ne(df["word"].shift()), time=None)
    )


def filter_signal(recording, passband=(0.5, 45), sfreq=int(1 / 0.004)):

    """
    Designes a digital Butterworth filter with the given parameters and applies it to the recorded signal.

    Parameters
    ----------
    recording: array_like
        The signal to be filtered.
    passband: tuple, optional
        Specifies the passband for the digital filter. Default is between 0.5 and 45 Hz.
    sfreq: int, optional
        Specifies the sampling frequency of the recording. Default is 250 Hz to match the sampling frequency of the Muse device.
    order: int, optional
        Order of the digital filter. Default is 10.

    Returns
    ----------
    out: ndarray
        The filtered signal.
    """

    sos = signal.butter(
        10,
        passband,
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    return signal.sosfiltfilt(sos, recording)
