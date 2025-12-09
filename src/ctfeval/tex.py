def number2tex(name, value, digits):
    return f"\\newcommand{{\\{name}}}{{{value:.{digits}f}}}\n"


def string2tex(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def section2tex(name):
    return f"% {name}\n\n"


def multirow(values, skip):
    """
    Add multirow to values when exporting pd.DataFrame to TeX.
    """
    new_values = []

    skip_counter = 0
    for value in values:
        if value not in skip:
            new_values.append(value)
            continue

        required_skips = skip[value]
        skip_counter += 1
        if skip_counter == 1:
            new_values.append(f"\\multirow{{{required_skips}}}{{*}}{{{value}}}")
            continue

        new_values.append("")
        if skip_counter == required_skips:
            skip_counter = 0

    return new_values
