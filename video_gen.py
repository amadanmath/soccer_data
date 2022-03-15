# TODO convert image manipulation to PIL
# https://stackoverflow.com/questions/50854235/how-to-draw-chinese-text-on-the-image-using-cv2-puttextcorrectly-pythonopen

from pathlib import Path
import json
from types import SimpleNamespace
import math
from datetime import timedelta
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

# pip install tqdm numpy pandas opencv-contrib-python==4.4.0.46

fps = 25.0
width, height = 1280, 720
font = cv2.FONT_HERSHEY_SIMPLEX

radius = 15
tscale = 0.6
team_colours = {
    1: ((0, 0, 0), (255, 128, 128)),
    2: ((0, 0, 0), (128, 128, 255)),
    3: ((255, 255, 255), (0, 0, 0)),
}
green = (0, 64, 0)
white = (255, 255, 255)
ball_colour = (255, 255, 255)
ball_radius = 10

htwidth, htheight = 5250, 3400
xmargin, ymargin = 20, 20

time_colour = (0, 192, 192)
time_tscale = 1

play_linger_frames = int(2 * fps)
num_slots = 6
action_font_size = 30
action_colour = (0, 192, 192)
bottom_space = action_font_size * 2

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
        game_dir = args.corpus_dir / args.game_id

        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName='/Library/Fonts/Arial Unicode.ttf', id=0)

        self.args = args
        self.tracking_data = pd.read_csv(game_dir / "tracking.csv")
        self.ball_data = pd.read_csv(game_dir / "ball.csv").set_index('フレーム番号')
        self.play_data = pd.read_csv(game_dir / "play.csv")
        self.grouped_play_data = self.play_data.groupby("フレーム番号")

        with (game_dir / "2_timestamp.json").open('rt') as r:
            half_data_raw = json.load(r)

        self.half_data=[
            range(half_data_raw["first_half_start"]["tracking_frame"], half_data_raw["first_half_end"]["tracking_frame"] + 1),
            range(half_data_raw["second_half_start"]["tracking_frame"], half_data_raw["second_half_end"]["tracking_frame"] + 1),
        ]

        self.action_displays = {}
        self.slots = [False] * num_slots

        self.make_background()


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


    def allocate_display_slot(self):
        slot = next((i for i, used in enumerate(self.slots) if not used), None)
        self.slots[slot] = True
        return slot


    def deallocate_display_slot(self, slot):
        self.slots[slot] = False


    def detect_current_actions(self):
        if self.frame_no in self.grouped_play_data.groups:
            play_data = self.grouped_play_data.get_group(self.frame_no)
            for ix in play_data.index:
                # phid: play history id
                phid = self.play_data["履歴No"][ix]
                attack_id = self.play_data["攻撃履歴No"][ix]
                start_phid = self.play_data["攻撃開始履歴No"][ix]
                end_phid = self.play_data["攻撃終了履歴No"][ix]
                death = play_linger_frames if phid == end_phid else None
                if attack_id in self.action_displays:
                    action_display = self.action_displays[attack_id]
                    action_display.ix = ix
                    action_display.death = death
                else:
                    slot = self.allocate_display_slot()
                    action_display = ActionDisplay(ix, death, slot)
                    self.action_displays[attack_id] = action_display

        attacks_to_delete = []
        for attack_id, action_display in self.action_displays.items():
            if action_display.age_check_dead():
                self.deallocate_display_slot(action_display.slot)
                attacks_to_delete.append(attack_id)
        for attack_id in attacks_to_delete:
            del self.action_displays[attack_id]


    def display_actions(self):
        for action_display in self.action_displays.values():
            ix = action_display.ix
            if action_display.death is None:
                factor = 1.0
            else:
                factor = action_display.death / play_linger_frames
            slot_width = (width - 2 * xmargin) / num_slots
            tx = int(xmargin + slot_width * action_display.slot)
            ty = int(height - bottom_space - ymargin)
            colour = [int(c * factor) for c in action_colour]
            action_name = self.play_data["アクション名"][ix]
            player_name = self.play_data["選手名"][ix]
            self.ft.putText(self.img, action_name, (tx, ty), action_font_size, colour, -1, cv2.LINE_AA, False)
            if isinstance(player_name, str):
                self.ft.putText(self.img, player_name, (tx, ty + action_font_size), action_font_size, colour, -1, cv2.LINE_AA, False)
                player_jersey = self.play_data["選手背番号"][ix]
                person_ix = next(ix for ix in self.frame_data.index if self.frame_data["背番号"][ix] == player_jersey)
                x = int(self.frame_data['座標X'][person_ix] / scale + cx)
                y = int(self.frame_data['座標Y'][person_ix] / scale + cy)
                # cv2.line(self.img, (tx, ty), (int(tx + slot_width), ty), colour, 1)
                cv2.line(self.img, (x, y), (int(tx + slot_width / 2), ty), colour, 1)
                cv2.circle(self.img, (x, y), radius + 2, colour, -1)


    def display_people(self):
        for index in self.frame_data.index:
            x = int(self.frame_data['座標X'][index] / scale + cx)
            y = int(self.frame_data['座標Y'][index] / scale + cy)
            fg_colour, bg_colour = team_colours[self.frame_data['ホームアウェイF'][index]]
            number = str(self.frame_data['背番号'][index])
            tw, th = cv2.getTextSize(number, font, tscale, 2)[0]
            tx = x - tw // 2
            ty = y + th // 2
            cv2.circle(self.img, (x, y), radius, bg_colour, -1)
            cv2.putText(self.img, number, (tx, ty), font, tscale, fg_colour, 2)


    def display_ball(self):
        try:
            x = int(self.ball_data['座標X'][self.frame_no] / scale + cx)
            y = int(self.ball_data['座標Y'][self.frame_no] / scale + cy)
            cv2.circle(self.img, (x, y), ball_radius, ball_colour, -1)
        except KeyError:
            pass


    def get_time(self):
        for index, frame_range in enumerate(self.half_data):
            if self.frame_no in frame_range:
                frame_offset = self.frame_no - frame_range.start
                seconds_offset = math.floor(frame_offset / fps)
                time = timedelta(seconds=seconds_offset)
                return index + 1, time
        return None, None


    def display_time(self):
        period, time = self.get_time()
        if period:
            time_text = f"{time}"
            ttw, tth = cv2.getTextSize(time_text, font, time_tscale, 2)[0]
            ttx = width - ttw - 10
            tty = tth + 5
            cv2.putText(self.img, time_text, (ttx, tty), font, time_tscale, time_colour, 2)
            period_text = f"{period}/2"
            ptw, pth = cv2.getTextSize(period_text, font, time_tscale, 2)[0]
            ptx = width - ptw - 10
            pty = pth + tty
            cv2.putText(self.img, period_text, (ptx, pty), font, time_tscale, time_colour, 2)


    def make_frame(self, frame_data):
        self.img = self.background.copy(order='K')
        self.frame_data = frame_data
        self.frame_no = frame_data['フレーム番号'][frame_data.first_valid_index()]

        self.detect_current_actions()
        self.display_actions()

        self.display_people()
        self.display_ball()
        self.display_time()
        return self.img


    def make_video(self):
        output_path = self.args.output_dir / (self.args.game_id + ".mp4")
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            it = self.tracking_data.groupby(['試合ID', 'フレーム番号'])
            if self.args.progress:
                it = tqdm(it)
            for out_frame_no, (frame, frame_data) in enumerate(it):
                if out_frame_no > fps * 120: break # DEBUG just 2m
                img = self.make_frame(frame_data)
                out.write(img)
        finally:
            out.release()

        return output_path




if __name__ == "__main__":
    args = SimpleNamespace(
        corpus_dir=Path(""),
        game_id="sample_game",
        output_dir=Path("out"),
        progress=True,
    )
    video_gen = VideoGen(args)
    output_path = video_gen.make_video()
    print(output_path)
