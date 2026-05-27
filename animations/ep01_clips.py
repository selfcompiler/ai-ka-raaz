"""
🎬 Episode 1 — Individual B-roll clips for editing
Each scene = one short clip (5-15 seconds) to insert while narrating.

Render all:  manim -pql ep01_clips.py
Render one:  manim -pql ep01_clips.py PhoneAutocomplete
HD render:   manim -pqh ep01_clips.py PhoneAutocomplete
"""
from manim import *


class Ep01_Clip1_PhoneAutocomplete(Scene):
    """INSERT AT 0:30 — When you say 'WhatsApp kholo, type karo Main aaj'"""
    def construct(self):
        typed = Text('"Main aaj"', font_size=52, color=WHITE)
        typed.shift(UP * 1)
        self.play(Write(typed), run_time=1.2)
        self.wait(0.3)

        arrow = Arrow(typed.get_bottom(), typed.get_bottom() + DOWN * 1, color=YELLOW)
        self.play(GrowArrow(arrow), run_time=0.4)

        suggestions = VGroup()
        for word in ["ghar", "office", "kha raha hoon"]:
            t = Text(word, font_size=32, color=GREEN)
            box = SurroundingRectangle(t, color=GREEN, corner_radius=0.15, buff=0.2)
            suggestions.add(VGroup(box, t))
        suggestions.arrange(RIGHT, buff=0.8)
        suggestions.next_to(arrow, DOWN, buff=0.5)

        for s in suggestions:
            self.play(FadeIn(s, scale=0.8), run_time=0.3)

        self.wait(1.5)


class Ep01_Clip2_NextWordBigText(Scene):
    """INSERT AT 1:30 — Big reveal: 'NEXT WORD KYA HOGA?'"""
    def construct(self):
        logos = VGroup(
            Text("ChatGPT", font_size=36, color=GREEN),
            Text("Claude", font_size=36, color=PURPLE),
            Text("Gemini", font_size=36, color=BLUE),
        ).arrange(RIGHT, buff=1)
        logos.shift(UP * 1.5)
        self.play(LaggedStart(*[FadeIn(l, scale=0.8) for l in logos], lag_ratio=0.2))
        self.wait(0.3)

        arrow = Arrow(logos.get_bottom(), logos.get_bottom() + DOWN * 1, color=YELLOW)
        self.play(GrowArrow(arrow), run_time=0.4)

        big = Text("NEXT WORD KYA HOGA?", font_size=52, color=YELLOW, weight=BOLD)
        big.next_to(arrow, DOWN, buff=0.5)
        self.play(Write(big), run_time=1)
        self.play(
            big.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=0.6
        )
        self.wait(1)


class Ep01_Clip3_CharacterPredict(Scene):
    """INSERT AT 2:30 — When explaining character-by-character prediction"""
    def construct(self):
        steps = [
            ("T", "h", "43%"),
            ("Th", "e", "65%"),
            ("The", "␣", "80%"),
            ("The ", "k", "12%"),
            ("The k", "i", "55%"),
            ("The ki", "n", "70%"),
            ("The kin", "g", "85%"),
        ]

        built = ""
        text_mob = Text("_", font_size=56, color=GRAY)
        text_mob.shift(UP * 0.5)
        prob_mob = Text("", font_size=28)
        prob_mob.shift(DOWN * 0.8)
        self.add(text_mob)

        for context, next_ch, prob in steps:
            built = context + next_ch
            display_ch = " " if next_ch == "␣" else next_ch

            new_text = Text(built + "_", font_size=56, color=WHITE)
            new_text.shift(UP * 0.5)

            new_prob = Text(
                f'"{context}" → "{display_ch}"  ({prob})',
                font_size=28, color=BLUE_C
            )
            new_prob.shift(DOWN * 0.8)

            self.play(
                Transform(text_mob, new_text),
                Transform(prob_mob, new_prob),
                run_time=0.5
            )
            self.wait(0.3)

        final = Text("Pattern. Bas Pattern!", font_size=40, color=YELLOW)
        final.shift(DOWN * 2)
        self.play(Write(final))
        self.wait(1)


class Ep01_Clip4_ScaleComparison(Scene):
    """INSERT AT 4:30 — Comparison table: our model vs GPT-4"""
    def construct(self):
        title = Text("Same Concept. Scale Alag.", font_size=38, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        small = Circle(radius=0.3, color=BLUE, fill_opacity=0.7)
        small.shift(LEFT * 3)
        small_label = VGroup(
            Text("Humara Model", font_size=22, color=BLUE),
            Text("4,225", font_size=20, color=GRAY),
        ).arrange(DOWN, buff=0.1)
        small_label.next_to(small, DOWN, buff=0.3)

        big = Circle(radius=2.2, color=RED, fill_opacity=0.3)
        big.shift(RIGHT * 1.5)
        big_label = VGroup(
            Text("GPT-4", font_size=22, color=RED),
            Text("1,800,000,000,000", font_size=20, color=GRAY),
        ).arrange(DOWN, buff=0.1)
        big_label.next_to(big, DOWN, buff=0.3)

        self.play(GrowFromCenter(small), FadeIn(small_label), run_time=0.6)
        self.wait(0.3)
        self.play(GrowFromCenter(big), FadeIn(big_label), run_time=1.2)
        self.wait(0.3)

        analogy = Text("Cycle vs Rocket — dono wheels pe chalte hain!", font_size=26, color=GREEN)
        analogy.to_edge(DOWN, buff=0.8)
        self.play(Write(analogy))
        self.wait(2)


class Ep01_Clip5_Cliffhanger(Scene):
    """INSERT AT 5:30 — Teaser for next episode"""
    def construct(self):
        words = VGroup(
            Text("cat = 42", font_size=36, color=WHITE),
            Text("dog = 51", font_size=36, color=WHITE),
            Text("rocket = 89", font_size=36, color=WHITE),
        ).arrange(DOWN, buff=0.4)
        words.shift(UP * 0.5)
        self.play(LaggedStart(*[Write(w) for w in words], lag_ratio=0.3))
        self.wait(0.5)

        q1 = Text("cat aur dog SIMILAR hain...", font_size=28, color=GREEN)
        q2 = Text("lekin cat aur rocket NAHI...", font_size=28, color=RED)
        q3 = Text("Computer ko kaise samjhaye?", font_size=30, color=YELLOW)
        q1.next_to(words, DOWN, buff=0.6)
        q2.next_to(q1, DOWN, buff=0.3)
        q3.next_to(q2, DOWN, buff=0.5)

        self.play(Write(q1))
        self.play(Write(q2))
        self.wait(0.3)
        self.play(Write(q3))
        self.wait(2)
