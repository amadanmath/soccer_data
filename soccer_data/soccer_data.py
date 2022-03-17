#!/usr/bin/env python

# pip install pandas openpyxl

from datetime import timedelta
from pathlib import Path
import json
import math

import pandas as pd

from typing import TYPE_CHECKING, Tuple, Optional, Union
if TYPE_CHECKING:
    from _typeshed import StrPath


fps = 25



class SoccerText:
    """Container for the generated commentary data

    Attributes
    ----------
    data: DataFrame
        commentary data
    data_gby_game: DataFrameGroupBy
        commentary data grouped by game
    """
    def __init__(self, path: 'StrPath'):
        """Loads commentary data from an XLSX file

        Parameters
        ----------
        path: StrPath
            Path to a XLSX file with commentary data
        """
        dtype = {
            '試合ID': str,
        }
        self.data = pd.read_excel(path, dtype={
            '節': 'int64',
            '試合ID': str,
            'ホーム': str,
            'アウェイ': str,
            '履歴No': 'int64',
            '前半:1後半:2': 'int64',
            'ハーフ時間（分）': 'int64',
            'コメント': str,
        })
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

    Attributes
    ----------
    game_id: str
        Game ID
    tracking: DataFrame
        Tracking data
    ball: DataFrame
        Ball tracking data
    play: DataFrame
        Play data
    text: DataFrame
        Commentary data
    tracking_gby_frame: DataFrameGroupBy
        Tracking data grouped by frame number
    ball_by_frame: DataFrame
        Ball tracking data indexed by frame number
    play_gby_frame: DataFrameGroupBy
        Play data grouped by frame number
    text_gby_frame: DataFrameGroupBy
        Commentary data grouped by frame number
    """
    def __init__(self, path: 'StrPath', soccer_text: Optional[SoccerText] = None) -> None:
        """Loads game data from a game data directory

        The game data directory is assumed to contain the files
        `tracking.csv`, `play.csv`, `ball.csv` and `2_timestamp.json`.

        Parameters
        ----------
        path: StrPath
            Path to directory with game data
        soccer_text: SoccerText, optional
            Commentary data
        """
        dtype = {
            '試合ID': str,
            '選手背番号': 'Int32',
        }
        path = Path(path).resolve()
        self.tracking = pd.read_csv(path / "tracking.csv", dtype={
            '試合ID': str,
            'フレーム番号': int,
            'ホームアウェイF': int,
            'システムターゲットID': int,
            '背番号': int,
            '座標X': int,
            '座標Y': int,
            '時速': float,
        })
        self.ball = pd.read_csv(path / "ball.csv", dtype={
            '試合ID': str,
            'フレーム番号': int,
            'ホームアウェイF': int,
            'システムターゲットID': int,
            '背番号': int,
            '座標X': int,
            '座標Y': int,
            '時速': float,
        })
        self.play = pd.read_csv(path / "play.csv", dtype={
            '試合ID': str,
            '履歴No': int,
            '攻撃履歴No': int,
            '攻撃連続No': int,
            '試合状態ID': int,
            'ハーフ開始相対時間': int,
            'アディショナルタイム': int,
            '時間帯': int,
            'シリーズNo': int,
            'ホームアウェイF': int,
            'チームID': int,
            'チーム名': str,
            '選手ID': int,
            '選手名': str,
            '選手背番号': 'Int32',
            'ポジションID': int,
            'アクションID': int,
            'アクション名': str,
            '流れID': int,
            'F_インプレー': int,
            'F_ボールタッチ': int,
            'F_成功': int,
            'F_プレー数１': int,
            'F_プレー数２': int,
            'F_プレー数３': int,
            '部位ID': int,
            '攻撃方向': int,
            'ボールＸ': float,
            'ボールＹ': float,
            'セル位置コード': str,
            '位置座標X': float,
            '位置座標Y': float,
            '縦位置': int,
            '敵陣F': int,
            'ペナルティエリアF': int,
            '横位置': int,
            '方向角度': float,
            '方向4': str,
            '方向8': str,
            '距離': float,
            '時間': float,
            '時間2': int,
            '攻撃開始履歴No': int,
            '攻撃終了履歴No': int,
            '攻撃経過時間': float,
            '攻撃経過プレー数': int,
            'F_ボールゲイン': int,
            'F_ボールロスト': int,
            '点差': int,
            '自スコア': int,
            '相手スコア': int,
            '相手チームID': int,
            'F_セットプレー': int,
            'F_セットプレー中': int,
            'F_ゴール': int,
            'F_シュート': int,
            'ゴールセル位置': int,
            'F_セーブ': int,
            'F_シュートGK以外': int,
            'F_ミスヒット': int,
            'ゴール角度': float,
            'ゴール距離': float,
            'Insゴール': int,
            'Insシュート': int,
            'GK選手ID': int,
            'F_アシスト': int,
            'F_シュートアシスト': int,
            'F_パス': int,
            'F_クロス': int,
            'F_スルーパス': int,
            'F_フリックオン': int,
            'パスToWho': int,
            'F_フィード': int,
            'フィードID': int,
            'タッチ種別': int,
            'タッチ種別名': str,
            'F_パス受け': int,
            'F_トラップ': int,
            'F_ドリブル': int,
            'F_ペナルティエリア進入': int,
            'F_ペナルティ脇進入': int,
            'F_30mライン進入': int,
            'F_こぼれ球奪取': int,
            'F_クリア': int,
            'F_インターセプト': int,
            'F_ブロック': int,
            'F_タックル': int,
            'F_タックル奪取': int,
            '空中戦F': int,
            '相手選手ID': int,
            'F_キャッチ': int,
            'F_ファンブル': int,
            'F_ハンドクリア': int,
            'F_ゴールキック': int,
            'F_コーナーキック': int,
            'F_直接フリーキック': int,
            'F_間接フリーキック': int,
            'F_ペナルティキック': int,
            'F_スローイン': int,
            'F_ファウル': int,
            'F_PK与えた': int,
            'F_被ファウル': int,
            'F_オフサイド': int,
            'F_イエローカード': int,
            'F_レッドカード': int,
            'F_タッチ': int,
            'F_ポスト・バー': int,
            'シュート公式F': int,
            'シュート公式アシストF': int,
            'アクション公式時間': int,
            'HOTZONE3-1': int,
            'HOTZONE1-3': int,
            'HOTZONE2-3': int,
            'HOTZONE3-3': int,
            'HOTZONE3-3_Detail': int,
            'HOTZONE3-2': int,
            'HOTZONE4-6': int,
            'HOTZONE4-6_Detail': int,
            'HOTZONE5-6_Detail': int,
            'HOTZONE5-6_pa': int,
            'HOTZONE6-9': int,
            'A1_履歴No': int,
            'A1_チームID': int,
            'A1_選手ID': int,
            'A1_ボールＸ': float,
            'A1_ボールＹ': float,
            'A1_攻撃方向': int,
            'A1_位置座標X': float,
            'A1_位置座標Y': float,
            'A1_アクションID': int,
            'A1_ホームアウェイF': int,
            'A1_ボールタッチF': int,
            'A1_ゴールF': int,
            'A1_シュートF': int,
            'A1_クロスF': int,
            'A1_パスF': int,
            'A1_空中戦F': int,
            'A1_縦位置': int,
            'A1_横位置': int,
            'A1_ペナルティエリアF': int,
            'A2_履歴No': int,
            'A2_チームID': int,
            'A2_選手ID': int,
            'A2_ボールＸ': float,
            'A2_ボールＹ': float,
            'A2_攻撃方向': int,
            'A2_アクションID': int,
            'B1_履歴No': int,
            'B1_チームID': int,
            'B1_選手ID': int,
            'B1_ボールＸ': float,
            'B1_ボールＹ': float,
            'B1_攻撃方向': int,
            'B1_アクションID': int,
            'B1_ホームアウェイF': int,
            'B1_クロスF': int,
            'B1_パスF': int,
            'B1_縦位置': int,
            'B1_横位置': int,
            'B1_方向4': str,
            'B1_距離': float,
            'B2_履歴No': int,
            'B2_チームID': int,
            'B2_選手ID': int,
            'B2_ボールＸ': float,
            'B2_ボールＹ': float,
            'B2_攻撃方向': int,
            'B2_アクションID': int,
            'パターン履歴No': int,
            'リリース履歴No': int,
            'A1リリース履歴No': int,
            '絶対時間秒数': float,
            'F_デュエル勝利': int,
            'フリーワード項目ID': int,
            'F_意図的なプレー終了': int,
            'フレーム番号': int,
        })

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

        self.tracking_gby_frame = self.tracking.groupby('フレーム番号')
        self.ball_by_frame = self.ball.set_index('フレーム番号')
        self.play_gby_frame = self.play.groupby("フレーム番号")
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
