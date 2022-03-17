#!/usr/bin/env python

# pip install numpy pandas openpyxl

from pathlib import Path
import json
from datetime import timedelta
import math

import numpy as np
import pandas as pd # type: ignore

from typing import TYPE_CHECKING, Tuple, Optional, Union
if TYPE_CHECKING:
    from _typeshed import StrPath


fps = 25


class SoccerText:
    """Container for the generated commentary data

    Parameters
    ----------
    path: StrPath
        Path to a XLSX file with commentary data

    Attributes
    ----------
    data: DataFrame
        commentary data
    data_gby_game: DataFrameGroupBy
        commentary data grouped by game
    """
    def __init__(self, path: 'StrPath'):
        dtype = {
            '試合ID': str,
        }
        self.data = pd.read_excel(path, dtype=dtype)
        self.data_gby_game = self.data.groupby('試合ID')


    def __getitem__(self, game_id: str) -> pd.DataFrame:
        """Gets only the rows relevant to one game

        Parameters
        ----------
        game_id: str
            Game ID

        Returns
        -------
        DataFrame
            A `DataFrame` with data that pertains to `game_id`
        """
        return self.data_gby_game.get_group(game_id)


class SoccerData:
    """Container for game data

    Parameters
    ----------
    path: StrPath
        Path to directory with game data
    soccer_text: SoccerText, optional
        Commentary data

    Attributes
    ----------
    game_id: str
        Game ID
    tracking: DataFrame
        Tracking data
    tracking_gby_frame: DataFrameGroupBy
        Tracking data grouped by frame number
    ball: DataFrame
        Ball tracking data
    ball_by_frame: DataFrame
        Ball tracking data indexed by frame number
    play: DataFrame
        Play data
    play_gby_frame: DataFrameGroupBy
        Play data grouped by frame number
    text: DataFrame
        Commentary data
    text_gby_frame: DataFrameGroupBy
        Commentary data grouped by frame number
    """
    def __init__(self, path: 'StrPath', soccer_text: Optional[SoccerText] = None) -> None:
        dtype = {
            '試合ID': str,
        }
        path = Path(path).resolve()
        self.tracking = pd.read_csv(path / "tracking.csv", dtype=dtype)
        self.tracking_gby_frame = self.tracking.groupby('フレーム番号')
        self.ball = pd.read_csv(path / "ball.csv", dtype=dtype)
        self.ball_by_frame = self.ball.set_index('フレーム番号')
        self.play = pd.read_csv(path / "play.csv", dtype=dtype)
        self.play_gby_frame = self.play.groupby("フレーム番号")

        with (path / "2_timestamp.json").open('rt') as r:
            half_data_raw = json.load(r)

        self._half = [
            range(half_data_raw["first_half_start"]["tracking_frame"], half_data_raw["first_half_end"]["tracking_frame"] + 1),
            range(half_data_raw["second_half_start"]["tracking_frame"], half_data_raw["second_half_end"]["tracking_frame"] + 1),
        ]
        self.game_id = self.tracking['試合ID'][0]

        self.text = soccer_text[self.game_id].copy() if soccer_text else pd.DataFrame()
        self.text['フレーム番号'] = [
            self.to_frame(self.text['前半:1後半:2'][ix], 60 * self.text['ハーフ時間（分）'][ix])
            for ix in self.text.index
        ]
        self.text_gby_frame = self.text.groupby('フレーム番号')


    def from_frame(self, frame_no: int) -> Tuple[int, timedelta]:
        """Converts frame number to time in period.

        Parameters
        ----------
        frame_no: int
            Frame number (FPS 25)

        Returns
        -------
        period: int
            Period (1 or 2)
            If the frame falls outside either period, 0
        time: timedelta
            Time relative to the start of the period
            If the frame falls outside either period, relative to start of tracking
        """
        for index, frame_range in enumerate(self._half):
            if frame_no in frame_range:
                frame_offset = frame_no - frame_range.start
                period = index + 1
                break
        else:
            frame_offset = self.tracking['フレーム番号'][0]
            period = 0
        seconds_offset = math.floor(frame_offset / fps)
        time = timedelta(seconds=seconds_offset)
        return period, time


    def to_frame(self, period: int, time: Union[timedelta, int]) -> int:
        """Converts period and time relative to it into frame number

        Parameters
        ----------

        period: int
            Period (1 or 2)
            If the frame falls outside either period, 0
        time: timedelta or int
            Time relative to the start of the period
            If integer, it is understood as seconds
            If the frame falls outside either period, relative to start of tracking

        Returns
        -------
        frame_no: int
            Frame number (FPS 25)
        """
        if isinstance(time, timedelta):
            seconds = time.total_seconds()
        else:
            seconds = time
        num_frames = int(seconds * fps)
        if period == 0:
            frame_offset = self.tracking['フレーム番号'][0]
        else:
            frame_offset = self._half[period - 1].start
        return frame_offset + num_frames



if __name__ == '__main__':
    text_path = Path('docs/テキスト速報データ_2021年J1_7節～10節_40試合.xlsx')
    text = SoccerText(text_path)
    data_path = Path('sample_game')
    data = SoccerData(data_path, text)
