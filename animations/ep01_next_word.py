"""
🎬 Episode 1 Animation: "AI Sochta Kaise Hai?"
Shows how AI predicts the next word/character — just like phone autocomplete.

Run: manim -pql ep01_next_word.py NextWordPrediction
"""
from manim import *

class NextWordPrediction(Scene):
    def construct(self):
        # --- Scene 1: Title ---
        title = Text("AI Sochta Kaise Hai?", font_size=56, color=YELLOW)
        subtitle = Text("Episode 1: The Secret of AI", font_size=28, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # --- Scene 2: Phone Autocomplete ---
        phone_title = Text("Phone Autocomplete", font_size=36, color=BLUE)
        phone_title.to_edge(UP, buff=0.5)
        self.play(Write(phone_title))

        typed = Text('"Main aaj"', font_size=44, color=WHITE)
        typed.shift(UP * 0.8)
        self.play(Write(typed), run_time=1)
        self.wait(0.5)

        arrow = Arrow(typed.get_bottom(), typed.get_bottom() + DOWN * 1.2, color=YELLOW)
        self.play(GrowArrow(arrow))

        suggestions = VGroup(
            Text("ghar", font_size=36, color=GREEN),
            Text("office", font_size=36, color=GREEN),
            Text("kha raha hoon", font_size=36, color=GREEN),
        ).arrange(RIGHT, buff=1.0)
        suggestions.next_to(arrow, DOWN, buff=0.4)

        for s in suggestions:
            box = SurroundingRectangle(s, color=GREEN, corner_radius=0.15, buff=0.2)
            s.add(box)
            self.play(FadeIn(s, scale=0.8), run_time=0.4)

        self.wait(1)

        reveal = Text("Yahi AI hai! Bas... bohot BADE scale pe.", font_size=32, color=YELLOW)
        reveal.to_edge(DOWN, buff=1)
        self.play(Write(reveal), run_time=1.5)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 3: Next Character Prediction ---
        pred_title = Text("AI = Next Token Prediction", font_size=40, color=YELLOW)
        pred_title.to_edge(UP, buff=0.5)
        self.play(Write(pred_title))

        chars_data = [
            ("T", "h", "43%"),
            ("Th", "e", "65%"),
            ("The", " ", "80%"),
            ("The ", "k", "12%"),
            ("The k", "i", "55%"),
            ("The ki", "n", "70%"),
            ("The kin", "g", "85%"),
        ]

        current_text = Text("", font_size=48)
        current_text.shift(UP * 0.5)

        prob_label = Text("", font_size=28, color=GRAY)
        prob_label.shift(DOWN * 0.5)

        for context, next_char, prob in chars_data:
            full = context + next_char

            new_text = Text(full, font_size=48, color=WHITE)
            new_text.shift(UP * 0.5)

            highlight = Text(next_char, font_size=48, color=GREEN)
            # Position the highlight at the last character position
            highlight.move_to(new_text.get_right() - RIGHT * highlight.width / 2)

            new_prob = Text(f'"{context}" → "{next_char}" ({prob})', font_size=28, color=BLUE_C)
            new_prob.shift(DOWN * 0.5)

            self.play(
                Transform(current_text, new_text) if current_text.text else FadeIn(new_text),
                FadeIn(new_prob) if not prob_label.text else Transform(prob_label, new_prob),
                run_time=0.6
            )
            current_text = new_text
            prob_label = new_prob
            self.wait(0.4)

        self.wait(1)

        box = SurroundingRectangle(
            VGroup(current_text, prob_label),
            color=YELLOW, corner_radius=0.2, buff=0.3
        )
        pattern_text = Text("Pattern. Bas Pattern!", font_size=36, color=YELLOW)
        pattern_text.next_to(box, DOWN, buff=0.5)
        self.play(Create(box), Write(pattern_text))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 4: Scale Comparison ---
        comp_title = Text("Same Concept. Scale Alag.", font_size=40, color=YELLOW)
        comp_title.to_edge(UP, buff=0.5)
        self.play(Write(comp_title))

        # Our model — small circle
        small = Circle(radius=0.4, color=BLUE, fill_opacity=0.6)
        small.shift(LEFT * 3 + DOWN * 0.5)
        small_label = Text("Humara Model\n4,225", font_size=22, color=WHITE)
        small_label.next_to(small, DOWN, buff=0.3)

        # GPT-4 — big circle
        big = Circle(radius=2.0, color=RED, fill_opacity=0.4)
        big.shift(RIGHT * 1.5 + DOWN * 0.5)
        big_label = Text("GPT-4\n1,800,000,000,000", font_size=22, color=WHITE)
        big_label.next_to(big, DOWN, buff=0.3)

        self.play(GrowFromCenter(small), Write(small_label))
        self.wait(0.5)
        self.play(GrowFromCenter(big), Write(big_label), run_time=1.5)
        self.wait(0.5)

        same = Text("Same Concept ✓", font_size=32, color=GREEN)
        same.to_edge(DOWN, buff=0.8)
        self.play(Write(same))
        self.wait(1)

        # Cycle vs Rocket analogy
        analogy = Text("🚲 Cycle vs 🚀 Rocket — dono wheels pe chalte hain!", font_size=28, color=GRAY)
        analogy.next_to(same, UP, buff=0.3)
        self.play(FadeIn(analogy, shift=UP * 0.3))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

        # --- Scene 5: Closing ---
        closing = Text("AI = Pattern Prediction", font_size=48, color=YELLOW)
        closing2 = Text("It doesn't THINK. It PREDICTS.", font_size=32, color=RED)
        closing2.next_to(closing, DOWN, buff=0.5)
        self.play(Write(closing), run_time=1.5)
        self.play(Write(closing2), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(closing), FadeOut(closing2))


class TokenizationFlow(Scene):
    """Bonus: Shows the tokenization pipeline for Episode 2"""
    def construct(self):
        title = Text("Tokenization: Text → Numbers", font_size=44, color=YELLOW)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Input text
        input_text = Text('"Hello"', font_size=48, color=WHITE)
        input_text.shift(UP * 1.5)
        self.play(Write(input_text))
        self.wait(0.5)

        # Arrow down
        arrow1 = Arrow(input_text.get_bottom(), input_text.get_bottom() + DOWN * 1, color=YELLOW)
        tok_label = Text("TOKENIZE", font_size=24, color=YELLOW)
        tok_label.next_to(arrow1, RIGHT, buff=0.3)
        self.play(GrowArrow(arrow1), Write(tok_label))

        # Character split
        chars = VGroup()
        char_list = ['H', 'e', 'l', 'l', 'o']
        for c in char_list:
            box = VGroup(
                Square(side_length=0.8, color=BLUE, fill_opacity=0.3),
                Text(c, font_size=36, color=WHITE)
            )
            chars.add(box)
        chars.arrange(RIGHT, buff=0.15)
        chars.shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(c, scale=0.5) for c in chars], lag_ratio=0.15))
        self.wait(0.5)

        # Arrow down to numbers
        arrow2 = Arrow(chars.get_bottom(), chars.get_bottom() + DOWN * 1, color=GREEN)
        enc_label = Text("ENCODE", font_size=24, color=GREEN)
        enc_label.next_to(arrow2, RIGHT, buff=0.3)
        self.play(GrowArrow(arrow2), Write(enc_label))

        # Numbers
        nums = VGroup()
        num_list = [20, 43, 50, 50, 53]
        for n in num_list:
            box = VGroup(
                Square(side_length=0.8, color=GREEN, fill_opacity=0.3),
                Text(str(n), font_size=30, color=GREEN)
            )
            nums.add(box)
        nums.arrange(RIGHT, buff=0.15)
        nums.next_to(arrow2, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(n, scale=0.5) for n in nums], lag_ratio=0.15))

        # Insight
        insight = Text("Computer sirf NUMBERS samajhta hai!", font_size=28, color=YELLOW)
        insight.to_edge(DOWN, buff=0.8)
        self.play(Write(insight))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])
