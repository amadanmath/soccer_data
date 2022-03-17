from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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

# pip install tqdm numpy pandas opencv-contrib-python==4.4.0.46



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
        self.data = SoccerData(args.corpus_dir / args.game_id, args.text)
        self.score = (0, 0)

        self.action_displays = []
        self.action_display_death = None

        self.ft = cv2.freetype.createFreeType2()
        self.ft.loadFontData(fontFileName='/Library/Fonts/Arial Unicode.ttf', id=0)

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
            # player_play_data = play_data[~play_data["選手名"].isnull()] # XXX
            ixs = self.data.play_gby_frame.groups[self.frame_no]

            if len(ixs):
                self.action_displays = []
                for slot, ix in enumerate(ixs):
                    player_name = self.data.play["選手名"][ix]
                    if not isinstance(player_name, str):
                        continue

                    # phid: play history id
                    phid = self.data.play["履歴No"][ix]
                    attack_id = self.data.play["攻撃履歴No"][ix]
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
            action_name = self.data.play["アクション名"][ix]
            player_name = self.data.play["選手名"][ix]
            play_team = self.data.play['ホームアウェイF'][ix]
            fg_colour, bg_colour = team_colours[play_team]
            team_colour = [int(c * factor) for c in bg_colour]
            self.ft.putText(self.img, action_name, (tx, ty), action_font_size, team_colour, -1, cv2.LINE_AA, False)

            if isinstance(player_name, str):
                player_jersey = self.data.play["選手背番号"][ix]
                player_jersey = "?" if math.isnan(player_jersey) else int(player_jersey)
                self.ft.putText(self.img, f"{player_jersey}　{player_name}", (tx, ty + action_font_size), player_font_size, team_colour, -1, cv2.LINE_AA, False)
                person_ix = next((person_ix for person_ix in self.frame_ixs if self.data.tracking["背番号"][person_ix] == player_jersey and self.data.tracking["ホームアウェイF"][person_ix] == play_team), None)
                if person_ix:
                    player_team = self.data.tracking['ホームアウェイF'][person_ix]
                    x = int(self.data.tracking['座標X'][person_ix] / scale + cx)
                    y = int(self.data.tracking['座標Y'][person_ix] / scale + cy)
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

            self.score = (self.data.play["自スコア"][ix], self.data.play["相手スコア"][ix])

        if self.action_display_death:
            self.action_display_death -= 1
            if not self.action_display_death:
                self.action_display_death = None
                self.action_displays = []


    def display_people(self):
        for ix in self.frame_ixs:
            x = int(self.data.tracking['座標X'][ix] / scale + cx)
            y = int(self.data.tracking['座標Y'][ix] / scale + cy)
            fg_colour, bg_colour = team_colours[self.data.tracking['ホームアウェイF'][ix]]
            number = str(self.data.tracking['背番号'][ix])
            tw, th = cv2.getTextSize(number, font, tscale, 2)[0]
            tx = x - tw // 2
            ty = y + th // 2
            cv2.circle(self.img, (x, y), radius, bg_colour, -1)
            cv2.putText(self.img, number, (tx, ty), font, tscale, fg_colour, 2)


    def display_ball(self):
        try:
            x = int(self.data.ball_by_frame['座標X'][self.frame_no] / scale + cx)
            y = int(self.data.ball_by_frame['座標Y'][self.frame_no] / scale + cy)
            cv2.circle(self.img, (x, y), ball_radius, ball_colour, -1)
        except KeyError:
            pass


    def display_stats(self, stats):
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
            score_texts = [str(self.score[0]), ":", str(self.score[1])]
            score_colours = [team_colours[1][1], time_colour, team_colours[2][1]]
            stws = []
            for score_text in score_texts:
                stw, sth = cv2.getTextSize(score_text, font, time_tscale, 2)[0]
                stws.append(stw)
            sty = sth + pty + 5
            stx = width - sum(stws) - 10
            for score_text, score_colour, stw in zip(score_texts, score_colours, stws):
                cv2.putText(self.img, score_text, (stx, sty), font, time_tscale, score_colour, 2)
                stx += stw



    def make_frame(self, frame_no, frame_ixs):
        self.img = self.background.copy(order='K')
        self.frame_no = frame_no
        self.frame_ixs = frame_ixs

        self.detect_current_actions()
        play_stats = self.display_actions()

        self.display_people()
        self.display_ball()
        self.display_stats(play_stats)
        return self.img


    def make_audio(self, duration):
        audio_output_path = self.args.output_dir / (self.args.game_id + ".mp3")
        zero_frame = self.data.tracking['フレーム番号'][0]
        audio_gen = AudioGen(duration, self.args.polly)

        for frame_no, ixs in self.data.text_gby_frame.groups.items():
            text = '\n'.join(
                self.data.text['コメント'][ix]
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
        output_path = self.args.output_dir / (self.args.game_id + ".mp4")
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            it = self.data.tracking_gby_frame.groups.items()
            if self.args.progress:
                it = tqdm(it)
            out_frame_no = 0
            for frame_no, frame_ixs in it:
                if args.from_frame is not None and out_frame_no < args.from_frame:
                    continue
                if args.till_frame is not None and out_frame_no >= args.till_frame:
                    break

                img = self.make_frame(frame_no, frame_ixs)
                out.write(img)

                out_frame_no += 1
        finally:
            out.release()

        if args.audio:
            audio_output_path = self.make_audio(out_frame_no / fps)

        if args.audio != "nomux":
            self.mux(output_path, audio_output_path)

        print("done")


        return output_path




if __name__ == "__main__":
    args = SimpleNamespace(
        text_path=Path("docs/テキスト速報データ_2021年J1_7節～10節_40試合.xlsx"),
        corpus_dir=Path(""),
        game_id="sample_game",
        output_dir=Path("out"),
        progress=True,
        audio=True,
        from_frame=None,
        till_frame=fps * 4 * 60,
        polly=Polly('mine', 'ap-northeast-1', 'polly-cache'),
    )
    args.text=SoccerText(args.text_path)
    video_gen = VideoGen(args)
    output_path = video_gen.make_video()
    print(output_path)
