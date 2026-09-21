"""Explicit bounded run profiles; the historical pilot ceiling stays default.

The larger profile is for a separately registered original-start experiment,
not a way to extend a live run or turn a saved-prefix diagnostic into a task SR.
"""


def validate_run_budget(args):
    profile = getattr(args, "budget_profile", "pilot")
    if profile == "pilot":
        limits = (96, 3072, 2400)
    elif profile == "fullstart192":
        if (args.mode != "agent" or args.harness != "grounded" or
                args.prefix != 0 or args.replay_prefix_spec is not None):
            raise ValueError("Full-start budget requires a zero-prefix grounded agent run")
        limits = (192, 6144, 7200)
    else:
        raise ValueError("Unknown registered budget profile")
    values = (args.max_decisions, args.max_controls, args.max_seconds)
    if (type(args.prefix) is not int or not 0 <= args.prefix <= 448 or
            any(type(value) is not int or not 1 <= value <= upper
                for value, upper in zip(values, limits))):
        raise ValueError("Registered run budget exceeded")
    return limits
