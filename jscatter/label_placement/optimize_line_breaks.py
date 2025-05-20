from math import ceil, sqrt
from typing import Optional

import numpy as np

from .text import Text


def optimize_line_breaks(
    text: str,
    text_measurer: Text,
    font_size: int,
    target_aspect_ratio: float,
    max_lines: Optional[int] = None,
    word_break_lookahead=3,
) -> str:
    """
    Add line breaks to optimize label aspect ratio.

    Parameters
    ----------
    text : str
        The label text
    label_type : str
        The type of the label (used to look up font info)
    label_value : str
        The value of the label (used to look up font info)
    target_aspect_ratio : float
        The target aspect ratio to optimize for
    max_lines : Optional[int]
        The maximum number of lines

    Notes
    -----
    Existing line breaks will be ignored by this function

    Returns
    -------
    str
        Text with line breaks added for optimal aspect ratio
    """

    # Skip if text is empty or too short
    if not text or len(text) < 5:
        return text

    words = text.split()
    single_line_text = ' '.join(words)

    if len(words) < 3:  # Need at least 3 words for meaningful line breaks
        return single_line_text

    # Get whitespace width and line height
    space_width = text_measurer.whitespace_width * font_size
    line_height = text_measurer.font.line_height * font_size

    word_widths = text_measurer.measure_words(single_line_text, font_size)

    single_line_width = sum(word_widths) + space_width * (len(words) - 1)
    single_line_ratio = single_line_width / line_height

    # Calculate the ideal number of lines to achieve target aspect ratio
    # Formula: (width / num_lines) / (num_lines * line_height) = target_ratio
    # Therefore: num_lines = sqrt(width / (target_ratio * line_height))
    ideal_num_lines = max(
        1, ceil(sqrt(single_line_width / (target_aspect_ratio * line_height)))
    )

    if max_lines is not None:
        max_lines = max(1, min(len(words), max_lines))
    else:
        max_lines = len(words)

    target_num_lines = min(ideal_num_lines, max_lines)

    # No need to process if target is 1 line
    if target_num_lines == 1:
        return single_line_text

    cum_word_widths = np.cumsum(word_widths) + np.arange(0, len(words)) * space_width

    # Line width to achieve target aspect ratio
    target_line_width = target_aspect_ratio * line_height * target_num_lines

    if target_line_width * target_num_lines < single_line_width:
        # Due to max_lines or breakpoint limitations we cannot achieve the
        # target aspect ratio with the number of lines. To have a enough space
        # to allocate all words in the targeted number of lines, we're going to
        # enlarge the target line width
        target_line_width = single_line_width / target_num_lines

    # Now distribute words across lines to match target width
    lines = []
    current_index = 0
    max_line_width = 0
    prev_cum_words_width = 0

    # Since we know that `cumulative_word_widths` is monotonically
    # increasing, all we need to do is to search where we would "insert"
    # a word with `i * target_line_width`, which represent the line break
    # we're looking for.
    for i in range(1, target_num_lines + 1):
        if current_index >= len(words):
            break

        # For the last line, use all remaining words
        if i == target_num_lines:
            break_index = len(words)

        else:
            ideal_break_index = max(
                current_index + 1,
                np.searchsorted(cum_word_widths, i * target_line_width, side='right'),
            )

            # The candidate is ideal_break - 1 (which is just below target)
            # and forward_window more candidates (which are above target)
            candidate_break_index_end = min(
                len(words), ideal_break_index + word_break_lookahead
            )

            # Evaluate all potential breakpoints in the window
            potential_break_indices = np.arange(
                ideal_break_index, candidate_break_index_end + 1
            )

            # Calculate difference between each break's width and the target width
            break_widths = (
                cum_word_widths[potential_break_indices - 1] - prev_cum_words_width
            )
            width_differences = np.abs(break_widths - target_line_width)

            # Find the break with minimum difference from target width
            best_break_idx = np.argmin(width_differences)
            break_index = potential_break_indices[best_break_idx]

        new_cum_words_width = cum_word_widths[break_index - 1]
        max_line_width = max(max_line_width, new_cum_words_width - prev_cum_words_width)
        lines.append(words[current_index:break_index])

        current_index = break_index
        prev_cum_words_width = new_cum_words_width

    # Combine into final text
    multi_line_text = '\n'.join([' '.join(line) for line in lines])

    # Verify the new aspect ratio is better
    multi_line_ratio = max_line_width / (len(lines) * line_height)

    # Only use the new text if it improves the aspect ratio
    if abs(multi_line_ratio - target_aspect_ratio) < abs(
        single_line_ratio - target_aspect_ratio
    ):
        return multi_line_text
    else:
        return single_line_text
