#!/usr/bin/env python

# pip install tqdm numpy pandas opencv-contrib-python==4.4.0.46

from datetime import timedelta, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import argparse
import json
import math
import subprocess
import sys

from tqdm import tqdm # type: ignore
import cv2 # type: ignore
import numpy as np
import pandas as pd # type: ignore

from .soccer_data import SoccerText, SoccerData
from .audio_gen import Polly, AudioGen



def dump(*args, **kwargs):
    pd.set_option("max_columns", None)
    pd.set_option("max_rows", None)
    print(*args, **kwargs)
    pd.reset_option("max_columns")
    pd.reset_option("max_rows")

def debugger():
    import pdb; pdb.set_trace()




fps = 25.0
width, height = 1280, 720
font = cv2.FONT_HERSHEY_SIMPLEX

radius = 15
tscale = 0.6
team_colours = {
    1: ((0, 0, 0), (255, 128, 128)),
    2: ((0, 0, 0), (128, 128, 255)),
    3: ((0, 0, 0), (192, 192, 192)),
}
green = (0, 64, 0)
white = (255, 255, 255)
ball_colour = (255, 255, 255)
ball_radius = 10

progress_colours = ((128, 128, 128), (0, 0, 255))

htwidth, htheight = 5250, 3400
xmargin, ymargin = 20, 20

time_colour = (0, 192, 192)
time_tscale = 1

play_linger_frames = int(5 * fps)
play_linger_decay_frames = int(4 * fps)
num_slots = 4
action_font_size = 30
player_font_size = 30
action_colour = (128, 255, 255)
bottom_space = action_font_size + player_font_size

scale = max(htwidth * 2 / (width - xmargin), htheight * 2 / (height - 2 * ymargin - bottom_space))
cx, cy = width / 2, (height - bottom_space - ymargin) / 2


class ActionDisplay:
    def __init__(self, ix, death, slot):
        self.ix = ix
        self.death = death
        self.slot = slot


    def age_check_dead(self):
        if self.death is None:
            pass
        elif self.death == 0:
            return True
        else:
            self.death -= 1
        return False


    def __repr__(self):
        return f"ActionDisplay({self.ix}, {self.death}, {self.slot})"



class VideoGen:
    def __init__(self, args):
        self.args = args
        self.data = SoccerData(args.game, args.text)

        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName='/Library/Fonts/Arial Unicode.ttf', id=0)

        self.reset_action_displays()
        self.make_background()


    def reset_action_displays(self):
        self.action_displays = []
        self.action_display_death = None


    def make_background(self):
        img = np.zeros((height, width, 3), np.uint8)
        hw = htwidth / scale
        hh = htheight / scale

        cv2.rectangle(img, (int(-hw + cx), int(-hh + cy)), (int(hw + cx), int(hh + cy)), green, -1)
        noise = np.random.random_sample(img.shape) * 0.4 + 0.8
        img = (img * noise).astype(np.uint8)

        # According to https://en.wikipedia.org/wiki/Football_pitch
        # field
        cv2.rectangle(img, (int(-hw + cx), int(-hh + cy)), (int(hw + cx), int(hh + cy)), white, 1)
        # half-way line
        cv2.line(img, (int(cx), int(-hh + cy)), (int(cx), int(hh + cy)), white, 1)

        # goal
        ghh = 732 / 2 / scale
        gw = 305 / scale
        cv2.rectangle(img, (int(-hw - gw + cx), int(-ghh + cy)), (int(-hw + cx), int(ghh + cy)), white, 1)
        cv2.rectangle(img, (int(hw + gw + cx), int(-ghh + cy)), (int(hw + cx), int(ghh + cy)), white, 1)

        # goal area
        ahh = 1932 / 2 / scale
        aw = 550 / scale
        cv2.rectangle(img, (int(-hw + aw + cx), int(-ahh + cy)), (int(-hw + cx), int(ahh + cy)), white, 1)
        cv2.rectangle(img, (int(hw - aw + cx), int(-ahh + cy)), (int(hw + cx), int(ahh + cy)), white, 1)

        # penalty area
        phh = 4030 / 2 / scale
        pw = 1650 / scale
        cv2.rectangle(img, (int(-hw + pw + cx), int(-phh + cy)), (int(-hw + cx), int(phh + cy)), white, 1)
        cv2.rectangle(img, (int(hw - pw + cx), int(-phh + cy)), (int(hw + cx), int(phh + cy)), white, 1)

        # centre circle
        cr = 915 / scale
        icr = int(cr)
        cv2.circle(img, (int(cx), int(cy)), icr, white, 1)

        # centre spot
        l = 20 / scale
        cv2.line(img, (int(cx - l), int(cy)), (int(cx + l), int(cy)), white, 1)

        # penalty spots
        l = 30 / scale
        psw = 1100 / scale
        cv2.line(img, (int(-hw + psw + cx), int(cy - l)), (int(-hw + psw + cx), int(cy + l)), white, 1)
        cv2.line(img, (int(hw - psw + cx), int(cy - l)), (int(hw - psw + cx), int(cy + l)), white, 1)

        # penalty circle
        a = math.acos((pw - psw) / cr) * 180 / math.pi
        cv2.ellipse(img, (int(-hw + psw + cx), int(cy)), (icr, icr), 0, -a, a, white, 1)
        cv2.ellipse(img, (int(hw - psw + cx), int(cy)), (icr, icr), 0, 180-a, 180+a, white, 1)

        # corners
        ior = int(100 / scale)
        cv2.ellipse(img, (int(-hw + cx), int(-hh + cy)), (ior, ior), 0, 0, 90, white, 1)
        cv2.ellipse(img, (int(-hw + cx), int(hh + cy)), (ior, ior), 0, 270, 360, white, 1)
        cv2.ellipse(img, (int(hw + cx), int(-hh + cy)), (ior, ior), 0, 90, 180, white, 1)
        cv2.ellipse(img, (int(hw + cx), int(hh + cy)), (ior, ior), 0, 180, 270, white, 1)

        # corner distance marks
        cdm = 915 / scale
        l = 20 / scale
        cv2.line(img, (int(-hw - l + cx), int(-hh + cdm + cy)), (int(-hw + cx), int(-hh + cdm + cy)), white, 1)
        cv2.line(img, (int(hw + l + cx), int(-hh + cdm + cy)), (int(hw + cx), int(-hh + cdm + cy)), white, 1)
        cv2.line(img, (int(-hw - l + cx), int(hh - cdm + cy)), (int(-hw + cx), int(hh - cdm + cy)), white, 1)
        cv2.line(img, (int(hw + l + cx), int(hh - cdm + cy)), (int(hw + cx), int(hh - cdm + cy)), white, 1)
        cv2.line(img, (int(-hw + cdm + cx), int(-hh - l + cy)), (int(-hw + cdm + cx), int(-hh + cy)), white, 1)
        cv2.line(img, (int(-hw + cdm + cx), int(hh + l + cy)), (int(-hw + cdm + cx), int(hh + cy)), white, 1)
        cv2.line(img, (int(hw - cdm + cx), int(-hh - l + cy)), (int(hw - cdm + cx), int(-hh + cy)), white, 1)
        cv2.line(img, (int(hw - cdm + cx), int(hh + l + cy)), (int(hw - cdm + cx), int(hh + cy)), white, 1)

        self.background = img


    # def allocate_display_slot(self):
    #     slot = next((i for i, used in enumerate(self.slots) if not used), None)
    #     self.slots[slot] = True
    #     print(slot, self.slots)
    #     return slot


    # def deallocate_display_slot(self, slot):
    #     self.slots[slot] = False


    def detect_current_actions(self):
        if self.frame_no in self.data.play_gby_frame.groups:
            # play_data = self.data.play_gby_frame.get_group(self.frame_no) # XXX
            # player_play_data = play_data[~play_data["?????????"].isnull()] # XXX
            ixs = self.data.play_gby_frame.groups[self.frame_no]

            if len(ixs):
                self.action_displays = []
                for slot, ix in enumerate(ixs):
                    player_name = self.data.play["?????????"][ix]
                    if not isinstance(player_name, str):
                        continue

                    # phid: play history id
                    phid = self.data.play["??????No"][ix]
                    attack_id = self.data.play["????????????No"][ix]
                    self.action_displays.append(ix)
                self.action_display_death = play_linger_frames
            else:
                self.action_displays = []
                self.action_display_death = None


    def display_actions(self):
        for slot, ix in enumerate(self.action_displays):
            if slot >= num_slots:
                print("Not enough slots", file=sys.stderr)
                continue
            if self.action_display_death > play_linger_decay_frames:
                factor = 1.0
            else:
                factor = self.action_display_death / play_linger_decay_frames
            slot_width = (width - 2 * xmargin) / num_slots
            tx = int(xmargin + slot_width * slot)
            ty = int(height - bottom_space - ymargin)
            colour = [int(c * factor) for c in action_colour]
            action_name = self.data.play["??????????????????"][ix]
            player_name = self.data.play["?????????"][ix]
            play_team = self.data.play['?????????????????????F'][ix]
            fg_colour, bg_colour = team_colours[play_team]
            team_colour = [int(c * factor) for c in bg_colour]
            self.ft.putText(self.img, action_name, (tx, ty), action_font_size, team_colour, -1, cv2.LINE_AA, False)

            if isinstance(player_name, str):
                player_jersey = self.data.play["???????????????"][ix]
                has_player = not pd.isna(player_jersey)
                player_jersey_display = player_jersey if has_player else "?"
                self.ft.putText(self.img, f"{player_jersey_display}???{player_name}", (tx, ty + action_font_size), player_font_size, team_colour, -1, cv2.LINE_AA, False)
                person_ix = has_player and next(
                    (
                        person_ix for person_ix in self.frame_ixs
                        if self.data.tracking["?????????"][person_ix] == player_jersey
                        and self.data.tracking["?????????????????????F"][person_ix] == play_team
                    ), None)
                if person_ix:
                    player_team = self.data.tracking['?????????????????????F'][person_ix]
                    x = int(self.data.tracking['??????X'][person_ix] / scale + cx)
                    y = int(self.data.tracking['??????Y'][person_ix] / scale + cy)
                    # cv2.line(self.img, (tx, ty), (int(tx + slot_width), ty), colour, 1)
                    # cv2.line(self.img, (x, y), (int(tx + slot_width / 2), ty), team_colour, 1)

                    star_points = 7
                    star_radius = radius + 5
                    star_slow = 8
                    #if star_radius <= x < width and star_radius <= y < height:
                    points = np.int32([
                        (x + star_radius * math.cos(a), y + star_radius * math.sin(a))
                        for a in (self.frame_no / star_slow + 4 * math.pi * i / star_points for i in range(star_points))
                    ])
                    cv2.fillPoly(self.img, [points], action_colour)
                    # cv2.circle(self.img, (x, y), radius + 3, action_colour, -1)

        if self.action_display_death:
            self.action_display_death -= 1
            if not self.action_display_death:
                self.action_display_death = None
                self.action_displays = []


    def display_people(self):
        for ix in self.frame_ixs:
            x = int(self.data.tracking['??????X'][ix] / scale + cx)
            y = int(self.data.tracking['??????Y'][ix] / scale + cy)
            fg_colour, bg_colour = team_colours[self.data.tracking['?????????????????????F'][ix]]
            number = str(self.data.tracking['?????????'][ix])
            tw, th = cv2.getTextSize(number, font, tscale, 2)[0]
            tx = x - tw // 2
            ty = y + th // 2
            cv2.circle(self.img, (x, y), radius, bg_colour, -1)
            cv2.putText(self.img, number, (tx, ty), font, tscale, fg_colour, 2)


    def display_ball(self):
        try:
            x = int(self.data.ball_by_frame['??????X'][self.frame_no] / scale + cx)
            y = int(self.data.ball_by_frame['??????Y'][self.frame_no] / scale + cy)
            cv2.circle(self.img, (x, y), ball_radius, ball_colour, -1)
        except KeyError:
            pass


    def display_stats(self):
        period, time = self.data.from_frame(self.frame_no)
        if period:
            time_text = f"{time}"
            ttw, tth = cv2.getTextSize(time_text, font, time_tscale, 2)[0]
            ttx = width - ttw - 10
            tty = tth + 5
            cv2.putText(self.img, time_text, (ttx, tty), font, time_tscale, time_colour, 2)
            period_text = f"{period}/2"
            ptw, pth = cv2.getTextSize(period_text, font, time_tscale, 2)[0]
            ptx = width - ptw - 10
            pty = pth + tty + 5
            cv2.putText(self.img, period_text, (ptx, pty), font, time_tscale, time_colour, 2)

            score = self.data.score(self.frame_no)
            team_ixs = [(0, 1), (1, 0)][period - 1]
            score_texts = [str(score[team_ixs[0]]), ":", str(score[team_ixs[1]])]
            score_colours = [team_colours[team_ixs[0] + 1][1], time_colour, team_colours[team_ixs[1] + 1][1]]
            stws = []
            for score_text in score_texts:
                stw, sth = cv2.getTextSize(score_text, font, time_tscale, 2)[0]
                stws.append(stw)
            sty = sth + pty + 5
            stx = width - sum(stws) - 10
            for score_text, score_colour, stw in zip(score_texts, score_colours, stws):
                cv2.putText(self.img, score_text, (stx, sty), font, time_tscale, score_colour, 2)
                stx += stw


    def display_progress_bar(self, complete):
        cv2.rectangle(self.img, (0, height - 1), (width - 1, height - 2), progress_colours[0], 1)
        cv2.rectangle(self.img, (0, height - 1), (int(complete * width) - 1, height - 2), progress_colours[1], 1)


    def make_frame(self, frame_no, frame_ixs):
        self.img = self.background.copy(order='K')
        self.frame_no = frame_no
        self.frame_ixs = frame_ixs

        self.detect_current_actions()
        self.display_actions()

        self.display_people()
        self.display_ball()
        self.display_stats()
        return self.img


    def make_audio(self, duration):
        audio_output_path = self.args.out / (self.args.game_id + ".mp3")
        zero_frame = self.data.tracking['??????????????????'][0]
        audio_gen = AudioGen(duration, self.args.polly)

        for frame_no, ixs in self.data.text_gby_frame.groups.items():
            text = '\n'.join(
                self.data.text['????????????'][ix]
                for ix in ixs
            )
            audio_position = (frame_no - zero_frame) / fps
            audio = self.args.polly.synthesize(text)
            audio_gen.add(audio_position, audio)

        audio_gen.get().export(audio_output_path, format="mp3")
        return audio_output_path


    def mux(self, video_path, audio_path):
        with TemporaryDirectory(dir=video_path.parent) as mux_dir:
            mux_path = Path(mux_dir) / video_path.name
            cmd = [
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel',
                'error',
                '-i',
                str(video_path),
                '-i',
                str(audio_path),
                '-map',
                '0:v',
                '-map',
                '1:a',
                '-codec:v',
                'libx264',
                '-preset',
                'medium',
                '-crf',
                '23',
                '-codec:a',
                'aac',
                '-b:a',
                '64k',
                '-shortest',
                str(mux_path),
            ]
            subprocess.run(cmd, check=True)
            mux_path.replace(video_path)
            audio_path.unlink()


    def make_video(self):
        output_path = self.args.out / (self.args.game_id + ".mp4")
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            first_frame = self.data.tracking["??????????????????"].min()
            last_frame = self.data.tracking["??????????????????"].max()
            num_frames = last_frame - first_frame + 1
            it = range(args.range[0] or 0, args.range[1] or num_frames)
            if not self.args.no_progress:
                it = tqdm(it)
            for frame_ix in it:
                frame_no = first_frame + frame_ix
                frame_ixs = self.data.tracking_gby_frame.groups.get(frame_no, [])
                img = self.make_frame(frame_no, frame_ixs)
                out.write(img)

        finally:
            out.release()

        if not args.no_audio:
            audio_output_path = self.make_audio(num_frames / fps)
            if not args.no_mux:
                self.mux(output_path, audio_output_path)

        return output_path


    def click_handler(self, evt, x, y, flags, params):
        if evt == cv2.EVENT_LBUTTONDOWN:
            self.frame_ix = int(x * self.num_frames / width)
            self.play_start_frame_ix = None


    def show_video(self):
        try:
            cv2.namedWindow(self.args.game_id)
            cv2.setMouseCallback(self.args.game_id, self.click_handler)

            frames = list(self.data.tracking_gby_frame.groups.keys())
            self.first_frame = self.data.tracking["??????????????????"].min()
            self.last_frame = self.data.tracking["??????????????????"].max()
            self.num_frames = self.last_frame - self.first_frame + 1
            self.frame_ix = 0
            frame_length = 1000 / fps
            self.play_start_frame_ix = None
            paused = False
            while True:
                if self.play_start_frame_ix is None:
                    play_start_dt = datetime.now()
                    self.play_start_frame_ix = self.frame_ix

                frame_start_dt = datetime.now()
                frame_no = self.first_frame + self.frame_ix
                frame_ixs = self.data.tracking_gby_frame.groups.get(frame_no, [])
                img = self.make_frame(frame_no, frame_ixs)
                self.display_progress_bar(self.frame_ix / self.num_frames)
                if not paused:
                    self.frame_ix += 1
                cv2.imshow(self.args.game_id, img)

                delta_frames = self.frame_ix - self.play_start_frame_ix
                actual_delta_ms = int((datetime.now() - play_start_dt).total_seconds() * 1000)
                ideal_delta_ms = int(1000 * delta_frames / fps)
                catch_up_ms = ideal_delta_ms - actual_delta_ms
                if catch_up_ms < 1:
                    catch_up_ms = 1
                key = cv2.waitKey(catch_up_ms)

                jump = None
                if key == 27: # esc
                    return
                elif key == 97: # a
                    jump = -5
                elif key == 100: # d
                    jump = 5
                elif key == 113: # q
                    jump = -60
                elif key == 101: # e
                    jump = 60
                elif key == 32: # space
                    paused = not paused

                if jump is not None:
                    self.reset_action_displays()
                    self.frame_ix += int(jump * fps)
                    if self.frame_ix < 0:
                        self.frame_ix = 0
                    self.play_start_frame_ix = None

                if self.frame_ix >= self.num_frames:
                    return

        finally:
            cv2.destroyWindow(self.args.game_id)


default_text_path = Path("/groups/gac50547/SoccerData/data_stadium/docs/??????????????????????????????_2021???J1_7??????10???_40??????.xlsx")
default_polly_cache = Path("~/.local/state/soccer_data/polly_cache").expanduser()
default_out_path = Path(".")
default_region = "ap-northeast-1"

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise SoccerData game")
    parser.add_argument("game", type=Path, help="the dir containing the game files")
    parser.add_argument("--out", "-o", type=Path, help="the output dir (if not supplied, visualise interactively")
    parser.add_argument("--text", "-t", type=Path, default=default_text_path, help="Path to the XLSX file with texts")
    parser.add_argument("--no-audio", "-A", action="store_true", help="Do not generate audio")
    parser.add_argument("--no-mux", "-M", action="store_true", help="Do not mux audio")
    parser.add_argument("--no-progress", "-P", action="store_true", help="Do not display progress")
    parser.add_argument("--range", default=":", help="Range in seconds `start:end` (default: whole game)")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--region", default=default_region, help="AWS region")
    parser.add_argument("--polly-cache-dir", type=Path, help="TTS cache dir", default=default_polly_cache)
    args = parser.parse_args()
    if ":" not in args.range:
        args.range += ":"
    args.range = tuple(int(float(x) * fps) if x else None for x in args.range.split(':'))
    args.polly_cache_dir.mkdir(parents=True, exist_ok=True)
    args.game_id = args.game.name
    if not args.no_audio:
        args.polly = Polly(args.profile, args.region, args.polly_cache_dir)
    return args



if __name__ == "__main__":
    args = parse_args()
    args.text=SoccerText(args.text)
    video_gen = VideoGen(args)
    if args.out is None:
        video_gen.show_video()
    else:
        output_path = video_gen.make_video()
        print(output_path)
