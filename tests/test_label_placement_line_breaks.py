from test_label_placement import sample_data

from jscatter.font import arial
from jscatter.label_placement import LabelPlacement
from jscatter.label_placement.optimize_line_breaks import optimize_line_breaks
from jscatter.label_placement.text import Text
from jscatter.label_placement.utils import remove_line_breaks


def test_optimize_line_breaks():
    """Test the optimization of line breaks to achieve a target aspect ratio."""

    # Create text measurer with Arial font
    text_measurer = Text(arial.regular)

    # Test case 1: Single line text should remain unchanged if short
    short_text = 'Short text'
    result = optimize_line_breaks(
        text=short_text,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=1.0,
    )
    assert result == short_text

    # Test case 2: Long single-line text with high aspect ratio should be split
    long_text = 'This is a very long text that should be split into multiple lines to achieve the target aspect ratio'
    result = optimize_line_breaks(
        text=long_text,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=1.0,
    )

    assert '\n' in result
    assert remove_line_breaks(result) == long_text

    # Test case 3: Ensure the resulting aspect ratio is closer to target
    original_width, original_height = text_measurer.measure(long_text, 12)
    original_ratio = original_width / original_height

    result_width, result_height = text_measurer.measure(result, 12)
    result_ratio = result_width / result_height

    # The optimized result should be closer to target ratio than original
    assert abs(result_ratio - 1.0) < abs(original_ratio - 1.0)

    # Test case 4: Max lines constraint
    result_limited = optimize_line_breaks(
        text=long_text,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=1.0,
        max_lines=2,
    )

    # Count the number of lines
    num_lines = result_limited.count('\n') + 1
    assert num_lines <= 2

    # Fewer lines then without the max_lines limitation
    assert result_limited.count('\n') < result.count('\n')

    # Test case 5: Very wide target aspect ratio
    result_wide = optimize_line_breaks(
        text=long_text,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=10.0,
    )

    # Should have fewer lines than the result with ratio=1.0
    assert result_wide.count('\n') < result.count('\n')

    # Test case 6: Very narrow target aspect ratio
    result_narrow = optimize_line_breaks(
        text=long_text,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=0.2,
    )

    # Should have more lines than the result with ratio=1.0
    assert result_narrow.count('\n') >= result.count('\n')

    # Test case 7: Text with a single long word
    single_word = 'Supercalifragilisticexpialidocious'
    result_single = optimize_line_breaks(
        text=single_word,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=1.0,
    )

    # Single word can't be split, should remain unchanged
    assert result_single == single_word

    # Test case 8: Text with custom target ratio
    custom_ratio = 2.5
    result_custom = optimize_line_breaks(
        text=long_text,
        text_measurer=text_measurer,
        font_size=12,
        target_aspect_ratio=custom_ratio,
    )

    custom_width, custom_height = text_measurer.measure(result_custom, 12)
    custom_result_ratio = custom_width / custom_height

    # Result ratio should be closer to custom target than original
    assert abs(custom_result_ratio - custom_ratio) < abs(original_ratio - custom_ratio)


def test_label_optimization_with_line_breaks(sample_data):
    """Test that labels are optimized with line breaks when target_aspect_ratio is set."""
    # Create a sample with long labels
    long_label_data = sample_data.copy()

    # Replace some category labels with long text
    long_label_data.loc[long_label_data['category'] == 'A', 'category'] = (
        'This is a very long label that should be broken into multiple lines'
    )
    long_label_data.loc[long_label_data['category'] == 'B', 'category'] = (
        'Another extremely long label which should get optimized with proper line breaks for better visualization'
    )

    # Create a label placer with aspect ratio set
    label_placer_with_ratio = LabelPlacement(
        data=long_label_data,
        by='category',
        x='x',
        y='y',
        target_aspect_ratio=1.0,  # Square aspect ratio
    )

    # Create another label placer with aspect ratio and max lines
    label_placer_with_max_lines = LabelPlacement(
        data=long_label_data,
        by='category',
        x='x',
        y='y',
        target_aspect_ratio=1.0,
        max_lines=2,  # Limit to 2 lines
    )

    # Create a control label placer without line break optimization
    label_placer_control = LabelPlacement(
        data=long_label_data,
        by='category',
        x='x',
        y='y',
    )

    # Process all label placers
    labels_with_ratio = label_placer_with_ratio.compute()
    labels_with_max_lines = label_placer_with_max_lines.compute()
    labels_control = label_placer_control.compute()

    # Find the long labels in each result
    long_label1_with_ratio = labels_with_ratio[
        labels_with_ratio['label'].str.startswith('This')
    ].iloc[0]
    long_label1_with_max_lines = labels_with_max_lines[
        labels_with_max_lines['label'].str.startswith('This')
    ].iloc[0]
    long_label1_control = labels_control[
        labels_control['label'].str.startswith('This')
    ].iloc[0]

    long_label2_with_ratio = labels_with_ratio[
        labels_with_ratio['label'].str.startswith('Another')
    ].iloc[0]
    long_label2_with_max_lines = labels_with_max_lines[
        labels_with_max_lines['label'].str.startswith('Another')
    ].iloc[0]
    long_label2_control = labels_control[
        labels_control['label'].str.startswith('Another')
    ].iloc[0]

    # Test that line breaks were added with target_aspect_ratio
    assert '\n' in long_label1_with_ratio['label']
    assert '\n' in long_label2_with_ratio['label']

    # Test that no line breaks were added without target_aspect_ratio
    assert '\n' not in long_label1_control['label']
    assert '\n' not in long_label2_control['label']

    # Test that max_lines constraint is respected
    line_count_label1 = long_label1_with_max_lines['label'].count('\n') + 1
    line_count_label2 = long_label2_with_max_lines['label'].count('\n') + 1
    assert line_count_label1 <= 2
    assert line_count_label2 <= 2

    # Test that aspect ratio is actually improved
    # Calculate aspect ratios
    text_measurer = Text(label_placer_with_ratio.font['category'])

    # Calculate aspect ratios for the first label
    width1_ratio, height1_ratio = text_measurer.measure(
        long_label1_with_ratio['label'], long_label1_with_ratio['font_size']
    )
    ratio1_with_optimization = width1_ratio / height1_ratio

    width1_control, height1_control = text_measurer.measure(
        long_label1_control['label'], long_label1_control['font_size']
    )
    ratio1_control = width1_control / height1_control

    # Check that the optimized ratio is closer to the target (1.0)
    assert abs(ratio1_with_optimization - 1.0) < abs(ratio1_control - 1.0)

    # Verify that the label with max_lines has fewer lines than the one without constraints
    unconstrained_line_count = long_label1_with_ratio['label'].count('\n') + 1
    assert line_count_label1 <= unconstrained_line_count


def test_existing_line_breaks_handling(sample_data):
    """Test that existing line breaks in labels are handled correctly."""
    # Create a sample with labels containing line breaks
    line_break_data = sample_data.copy()

    # Replace some category labels with text that already has line breaks
    line_break_data.loc[line_break_data['category'] == 'A', 'category'] = (
        'Line\nBreak\nAlready\nExists'
    )
    line_break_data.loc[line_break_data['category'] == 'B', 'category'] = (
        'Another\nPre-formatted\nMulti-line Label'
    )
    line_break_data.loc[line_break_data['category'] == 'C', 'category'] = (
        'This one has\nfewer breaks'
    )

    # Create label placers with different configurations
    # 1. Without aspect ratio optimization (should maintain existing breaks)
    label_placer_no_opt = LabelPlacement(
        data=line_break_data,
        by='category',
        x='x',
        y='y',
    )

    # 2. With aspect ratio optimization (should ignore existing breaks)
    label_placer_with_opt = LabelPlacement(
        data=line_break_data,
        by='category',
        x='x',
        y='y',
        target_aspect_ratio=1.0,
    )

    # Additional test for font properties with existing line breaks
    label_placer_with_props = LabelPlacement(
        data=line_break_data,
        by='category',
        x='x',
        y='y',
        color={
            # Use category name without line breaks for mapping
            'category:Line Break Already Exists': 'red',
            'category:Another Pre-formatted Multi-line Label': 'blue',
        },
        size={
            # Use category name without line breaks for mapping
            'category:Line Break Already Exists': 20,
            'category:Another Pre-formatted Multi-line Label': 16,
        },
    )

    # Process all label placers
    labels_no_opt = label_placer_no_opt.compute()
    labels_with_opt = label_placer_with_opt.compute()
    labels_with_props = label_placer_with_props.compute()

    # 1. Test that existing line breaks are maintained when no optimization is applied
    label_a_no_opt = labels_no_opt[labels_no_opt['label'].str.startswith('Line')].iloc[
        0
    ]
    assert label_a_no_opt['label'] == 'Line\nBreak\nAlready\nExists'

    label_b_no_opt = labels_no_opt[
        labels_no_opt['label'].str.startswith('Another')
    ].iloc[0]
    assert label_b_no_opt['label'] == 'Another\nPre-formatted\nMulti-line Label'

    # 2. Test that existing line breaks are ignored during aspect ratio optimization
    # The optimized text should contain line breaks, but likely at different positions
    label_a_with_opt = labels_with_opt[
        labels_with_opt['label'].str.startswith('Line')
    ].iloc[0]

    # The optimized text should be different from the original but have the same content when line breaks are removed
    assert label_a_with_opt['label'] != 'Line\nBreak\nAlready\nExists'
    assert remove_line_breaks(label_a_with_opt['label']) == remove_line_breaks(
        'Line\nBreak\nAlready\nExists'
    )

    # 3. Test that mapped properties work correctly with labels containing line breaks
    label_a_with_props = labels_with_props[
        labels_with_props['label'].str.startswith('Line')
    ].iloc[0]
    label_b_with_props = labels_with_props[
        labels_with_props['label'].str.startswith('Another')
    ].iloc[0]

    # Font color should be applied correctly despite line breaks in original label
    assert label_a_with_props['font_color'] == '#ff0000'  # red
    assert label_b_with_props['font_color'] == '#0000ff'  # blue

    # Font size should be applied correctly despite line breaks in original label
    assert label_a_with_props['font_size'] == 20
    assert label_b_with_props['font_size'] == 16

    # Test the internal mapping structures to verify line breaks are correctly handled
    assert 'category:Line Break Already Exists' in label_placer_with_props.color
    assert (
        'category:Another Pre-formatted Multi-line Label'
        in label_placer_with_props.color
    )

    # Verify the text measurer can handle texts with line breaks correctly
    text_measurer = Text(arial.regular)
    width_with_breaks, height_with_breaks = text_measurer.measure(
        'Line\nBreak\nAlready\nExists', 12
    )
    width_without_breaks, height_without_breaks = text_measurer.measure(
        'Line Break Already Exists', 12
    )

    # Height should be greater with line breaks
    assert height_with_breaks > height_without_breaks
    # Width should be less with line breaks (stacked text is narrower)
    assert width_with_breaks < width_without_breaks
