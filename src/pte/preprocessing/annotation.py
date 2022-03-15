"""Module for annotating Raw objects and other data."""
import mne


def annotate_trials(
    raw: mne.io.BaseRaw, keyword: str = "EMG", inplace: bool = False
) -> mne.io.BaseRaw:
    """Create squared data (0s and 1s) from events and add to Raw object.

    Parameters
    ----------
    raw : MNE Raw object
        The MNE Raw object for this function to modify.
    keyword : str
        Keyword for the events to be added.
    inplace : bool. Default: False
        Set to True if Raw object should be modified in place.

    Returns
    -------
    raw : MNE Raw object
        The Raw object containing the added squared channel.
    """
    if not inplace:
        raw = raw.copy()

    raw.plot(
        scalings="auto",
        block=True,
        title=(
            f"Please annotate {keyword} onset and {keyword} end with the label"
            f" `{keyword}`."
        ),
    )

    events_emg, _ = mne.events_from_annotations(
        raw, event_id={f"{keyword}": 1}
    )
    events_emg[1::2, 2] = -1
    annotations_emg = mne.annotations_from_events(
        events_emg,
        raw.info["sfreq"],
        event_desc={1: f"{keyword}_onset", -1: f"{keyword}_end"},
    )

    raw.set_annotations(annotations_emg)

    raw.plot(
        scalings="auto",
        block=True,
        title=f"{keyword} onset and end annotated. Please check annotations.",
    )
    return raw
