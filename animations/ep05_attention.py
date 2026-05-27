"""
🎬 Episode 5 Animation: "AI Ka Sabse Powerful Weapon"
Shows how Attention lets words look at each other to understand context.

Run: manim -pql ep05_attention.py AttentionExplained
"""
from manim import *
import numpy as np

class AttentionExplained(Scene):
    def construct(self):
        # --- Scene 1: Title ---
        title = Text("AI Ka Sabse Powerful Weapon", font_size=48, color=YELLOW)
        subtitle = Text("Episode 5: ATTENTION", font_size=30, color=RED)
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # --- Scene 2: The Problem ---
        sentence = '"The animal didn\'t cross the street because IT was too tired."'
        sent_text = Text(sentence, font_size=24, color=WHITE)
        sent_text.to_edge(UP, buff=1)
        self.play(Write(sent_text), run_time=2)

        question = Text('"IT" ka matlab kya hai?  Animal  ya  Street?', font_size=30, color=YELLOW)
        question.next_to(sent_text, DOWN, buff=0.8)
        self.play(Write(question))
        self.wait(1.5)

        answer = Text("ATTENTION solves this!", font_size=34, color=GREEN)
        answer.next_to(question, DOWN, buff=0.5)
        self.play(Write(answer))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 3: Words Looking at Each Other ---
        attn_title = Text("Attention: Har word BAAKI words ko dekhta hai", font_size=30, color=YELLOW)
        attn_title.to_edge(UP, buff=0.5)
        self.play(Write(attn_title))

        words = ["The", "animal", "didn't", "cross", "because", "IT", "was", "tired"]
        word_mobs = VGroup()
        for w in words:
            t = Text(w, font_size=28, color=WHITE)
            box = SurroundingRectangle(t, color=BLUE_D, corner_radius=0.1, buff=0.15)
            group = VGroup(box, t)
            word_mobs.add(group)
        word_mobs.arrange(RIGHT, buff=0.3)
        word_mobs.shift(UP * 0.5)
        self.play(LaggedStart(*[FadeIn(w, scale=0.8) for w in word_mobs], lag_ratio=0.1))
        self.wait(0.5)

        # Highlight "IT"
        it_idx = 5
        it_box = word_mobs[it_idx]
        self.play(it_box[0].animate.set_color(YELLOW), run_time=0.5)

        # Attention arrows from IT to other words
        attention_weights = {
            0: 0.02,  # The
            1: 0.70,  # animal — HIGH
            2: 0.03,  # didn't
            3: 0.05,  # cross
            4: 0.05,  # because
            6: 0.10,  # was
            7: 0.05,  # tired
        }

        arrows = VGroup()
        weight_labels = VGroup()
        for idx, weight in attention_weights.items():
            opacity = weight * 1.4
            color = GREEN if weight > 0.3 else BLUE if weight > 0.08 else GRAY
            arrow = CurvedArrow(
                it_box.get_bottom() + DOWN * 0.1,
                word_mobs[idx].get_bottom() + DOWN * 0.1,
                angle=-TAU/6 if idx < it_idx else TAU/6,
                color=color,
                stroke_opacity=max(opacity, 0.2),
                stroke_width=2 + weight * 8,
            )
            pct = Text(f"{weight:.0%}", font_size=16, color=color)
            pct.next_to(word_mobs[idx], DOWN, buff=0.6)
            arrows.add(arrow)
            weight_labels.add(pct)

        self.play(
            LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.15),
            LaggedStart(*[FadeIn(l) for l in weight_labels], lag_ratio=0.15),
            run_time=2
        )

        insight = Text('"IT" ne 70% attention "animal" ko diya!', font_size=28, color=GREEN)
        insight.to_edge(DOWN, buff=0.8)
        self.play(Write(insight))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 4: Party Analogy (Q, K, V) ---
        party_title = Text("Attention = Party Mein Sahi Insaan Dhundna", font_size=32, color=YELLOW)
        party_title.to_edge(UP, buff=0.5)
        self.play(Write(party_title))

        steps = [
            ("Q", "QUERY", "\"Mujhe kya chahiye?\"", BLUE),
            ("K", "KEY", "\"Har insaan ka tag — yeh kya jaanta hai?\"", GREEN),
            ("V", "VALUE", "\"Uski actual info lo\"", RED),
        ]

        step_mobs = VGroup()
        for letter, name, desc, color in steps:
            big_letter = Text(letter, font_size=72, color=color, weight=BOLD)
            name_text = Text(name, font_size=28, color=color)
            desc_text = Text(desc, font_size=22, color=GRAY)
            name_text.next_to(big_letter, RIGHT, buff=0.3)
            desc_text.next_to(VGroup(big_letter, name_text), DOWN, buff=0.2)
            step_mobs.add(VGroup(big_letter, name_text, desc_text))

        step_mobs.arrange(DOWN, buff=0.6)
        step_mobs.shift(DOWN * 0.2)

        for s in step_mobs:
            self.play(FadeIn(s, shift=RIGHT * 0.5), run_time=0.8)
            self.wait(0.5)

        formula_box = VGroup(
            Text("Attention(Q, K, V) = softmax(Q × K", font_size=26, color=WHITE),
            Text("T", font_size=18, color=WHITE).shift(UP * 0.15),
            Text(" / √d) × V", font_size=26, color=WHITE),
        ).arrange(RIGHT, buff=0.05)
        formula_box.to_edge(DOWN, buff=0.6)
        box = SurroundingRectangle(formula_box, color=YELLOW, corner_radius=0.15, buff=0.2)
        self.play(Write(formula_box), Create(box))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 5: Multi-Head ---
        mh_title = Text("Multi-Head Attention: Team of Experts", font_size=36, color=YELLOW)
        mh_title.to_edge(UP, buff=0.5)
        self.play(Write(mh_title))

        heads = [
            ("Head 1", "Meaning\nconnection", BLUE),
            ("Head 2", "Grammar\nconnection", GREEN),
            ("Head 3", "Position\nconnection", RED),
            ("Head 4", "Emotional\ntone", PURPLE),
        ]

        head_mobs = VGroup()
        for name, desc, color in heads:
            circle = Circle(radius=0.7, color=color, fill_opacity=0.3)
            name_text = Text(name, font_size=22, color=color, weight=BOLD)
            desc_text = Text(desc, font_size=16, color=WHITE)
            name_text.move_to(circle.get_center() + UP * 0.2)
            desc_text.move_to(circle.get_center() + DOWN * 0.2)
            head_mobs.add(VGroup(circle, name_text, desc_text))

        head_mobs.arrange(RIGHT, buff=0.5)

        self.play(LaggedStart(*[GrowFromCenter(h) for h in head_mobs], lag_ratio=0.2))
        self.wait(0.5)

        gpt_note = Text("GPT-4: 96 heads! 96 different perspectives.", font_size=26, color=YELLOW)
        gpt_note.to_edge(DOWN, buff=1)
        self.play(Write(gpt_note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Final ---
        final = Text("Attention = Words Talking to Each Other", font_size=38, color=YELLOW)
        final2 = Text("This is why AI understands CONTEXT.", font_size=28, color=GREEN)
        final2.next_to(final, DOWN, buff=0.5)
        self.play(Write(final), run_time=1.5)
        self.play(Write(final2), run_time=1)
        self.wait(2)
