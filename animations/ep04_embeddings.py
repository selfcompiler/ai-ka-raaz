"""
🎬 Episode 4 Animation: "Words Ka Solar System"
Shows how words live in a space where similar words cluster together.

Run: manim -pql ep04_embeddings.py EmbeddingsUniverse
"""
from manim import *
import numpy as np

class EmbeddingsUniverse(Scene):
    def construct(self):
        # --- Scene 1: Title ---
        title = Text("Words Ka Solar System", font_size=52, color=YELLOW)
        subtitle = Text("Episode 4: Embeddings", font_size=28, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # --- Scene 2: The Problem ---
        problem_title = Text("Problem: Numbers Mein Meaning Nahi", font_size=36, color=RED)
        problem_title.to_edge(UP, buff=0.5)
        self.play(Write(problem_title))

        words_nums = VGroup(
            Text("cat = 42", font_size=36, color=WHITE),
            Text("dog = 51", font_size=36, color=WHITE),
            Text("rocket = 89", font_size=36, color=WHITE),
        ).arrange(DOWN, buff=0.5)
        words_nums.shift(DOWN * 0.3)
        self.play(LaggedStart(*[Write(w) for w in words_nums], lag_ratio=0.3))
        self.wait(0.5)

        question = Text("42 aur 51 mein kya similarity hai? KUCH NAHI!", font_size=28, color=YELLOW)
        question.to_edge(DOWN, buff=1)
        self.play(Write(question))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 3: Embedding = List of Numbers ---
        sol_title = Text("Solution: Ek number nahi — LIST of numbers!", font_size=34, color=GREEN)
        sol_title.to_edge(UP, buff=0.5)
        self.play(Write(sol_title))

        embed_data = [
            ("cat", [0.2, 0.8, 0.1], BLUE),
            ("dog", [0.3, 0.7, 0.2], BLUE),
            ("car", [0.9, 0.1, 0.8], RED),
        ]

        entries = VGroup()
        for word, vec, color in embed_data:
            vec_str = f"[{vec[0]}, {vec[1]}, {vec[2]}]"
            line = Text(f"{word:>6} = {vec_str}", font_size=34, color=color)
            entries.add(line)
        entries.arrange(DOWN, buff=0.5)

        self.play(Write(entries[0]), run_time=0.8)
        self.wait(0.3)
        self.play(Write(entries[1]), run_time=0.8)

        close_arrow = Arrow(
            entries[0].get_right() + RIGHT * 0.3,
            entries[1].get_right() + RIGHT * 0.3,
            color=GREEN, buff=0.1
        )
        close_label = Text("CLOSE!", font_size=24, color=GREEN)
        close_label.next_to(close_arrow, RIGHT, buff=0.2)
        self.play(GrowArrow(close_arrow), Write(close_label))
        self.wait(0.5)

        self.play(Write(entries[2]), run_time=0.8)
        far_label = Text("FAR!", font_size=24, color=RED)
        far_label.next_to(entries[2], RIGHT, buff=0.5)
        self.play(Write(far_label))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 4: Words Floating in Space ---
        space_title = Text("Words in Space", font_size=40, color=YELLOW)
        space_title.to_edge(UP, buff=0.3)
        self.play(Write(space_title))

        np.random.seed(42)
        word_positions = {
            "king":   np.array([-2.5,  1.5, 0]),
            "queen":  np.array([-1.5,  1.8, 0]),
            "prince": np.array([-2.0,  1.0, 0]),
            "cat":    np.array([ 2.0,  1.0, 0]),
            "dog":    np.array([ 2.8,  0.8, 0]),
            "fish":   np.array([ 2.5,  1.5, 0]),
            "run":    np.array([ 0.0, -1.5, 0]),
            "walk":   np.array([ 0.8, -1.8, 0]),
            "jump":   np.array([-0.5, -1.2, 0]),
            "car":    np.array([-2.5, -1.5, 0]),
            "rocket": np.array([-1.5, -2.0, 0]),
        }

        colors = {
            "king": PURPLE, "queen": PURPLE, "prince": PURPLE,
            "cat": BLUE, "dog": BLUE, "fish": BLUE,
            "run": GREEN, "walk": GREEN, "jump": GREEN,
            "car": RED, "rocket": RED,
        }

        dots = VGroup()
        labels = VGroup()
        for word, pos in word_positions.items():
            dot = Dot(pos, radius=0.12, color=colors[word])
            label = Text(word, font_size=22, color=colors[word])
            label.next_to(dot, DOWN, buff=0.15)
            dots.add(dot)
            labels.add(label)

        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.08),
            LaggedStart(*[FadeIn(l, scale=0.5) for l in labels], lag_ratio=0.08),
            run_time=2
        )
        self.wait(0.5)

        # Highlight clusters
        royalty_circle = Circle(radius=1.2, color=PURPLE, stroke_opacity=0.5)
        royalty_circle.move_to(np.array([-2.0, 1.4, 0]))
        royalty_label = Text("Royalty", font_size=20, color=PURPLE)
        royalty_label.next_to(royalty_circle, UP, buff=0.1)

        animal_circle = Circle(radius=1.1, color=BLUE, stroke_opacity=0.5)
        animal_circle.move_to(np.array([2.4, 1.1, 0]))
        animal_label = Text("Animals", font_size=20, color=BLUE)
        animal_label.next_to(animal_circle, UP, buff=0.1)

        action_circle = Circle(radius=1.2, color=GREEN, stroke_opacity=0.5)
        action_circle.move_to(np.array([0.1, -1.5, 0]))
        action_label = Text("Actions", font_size=20, color=GREEN)
        action_label.next_to(action_circle, DOWN, buff=0.1)

        self.play(
            Create(royalty_circle), Write(royalty_label),
            Create(animal_circle), Write(animal_label),
            Create(action_circle), Write(action_label),
            run_time=1.5
        )

        insight = Text("AI ne KHUD seekha ki related words paas hain!", font_size=26, color=YELLOW)
        insight.to_edge(DOWN, buff=0.5)
        self.play(Write(insight))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 5: The Magic — King - Man + Woman = Queen ---
        magic_title = Text("Embedding Magic", font_size=40, color=YELLOW)
        magic_title.to_edge(UP, buff=0.5)
        self.play(Write(magic_title))

        equation_parts = [
            Text("King", font_size=44, color=PURPLE),
            Text(" - ", font_size=44, color=WHITE),
            Text("Man", font_size=44, color=BLUE),
            Text(" + ", font_size=44, color=WHITE),
            Text("Woman", font_size=44, color=PINK),
            Text(" = ", font_size=44, color=WHITE),
            Text("???", font_size=44, color=YELLOW),
        ]
        equation = VGroup(*equation_parts).arrange(RIGHT, buff=0.15)

        for i, part in enumerate(equation_parts):
            self.play(Write(part), run_time=0.3)
        self.wait(1)

        answer = Text("Queen!", font_size=52, color=GOLD)
        answer.move_to(equation_parts[-1].get_center())
        self.play(
            Transform(equation_parts[-1], answer),
            Flash(answer, color=GOLD, num_lines=12),
            run_time=1
        )
        self.wait(0.5)

        explain = Text(
            "Numbers mein MEANING hai!",
            font_size=32, color=YELLOW
        )
        explain.next_to(equation, DOWN, buff=1)
        self.play(Write(explain))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Final ---
        final = Text("Embedding = Word ka Address in Space", font_size=38, color=YELLOW)
        final2 = Text("Similar words → Close addresses", font_size=30, color=GREEN)
        final2.next_to(final, DOWN, buff=0.5)
        self.play(Write(final), run_time=1.5)
        self.play(Write(final2), run_time=1)
        self.wait(2)
