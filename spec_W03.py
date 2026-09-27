"""W03 - variables and functions, using engineering formulas.

20 marks. q1-q3 visible (10 marks), q4-q5 hidden (10 marks). The hidden
checks live in hidden/spec_W03_hidden.py and are never handed out. This
file IS handed out, so assume every student reads it.

q3 has a per-cohort variant: each cohort writes a different function, taken
from its own discipline, and the task sheet for that cohort names it. The
grader picks the variant from the COHORT variable set at the top of the
notebook, or from Grader("W03", cohort="CML3112").

    CML3111  Electrical / Mechatronics & Robotics / Communications
    CML3112  Architecture / Civil
    CML3113  Mining / Petroleum

No pandas and no numpy are needed for this week. Plain Python only.
"""

ASSIGNMENT = "W03"
TITLE = "variables and functions"
DEFAULT_COHORT = "CML3111"
QUESTION_ORDER = ["q1", "q2", "q3", "q4", "q5"]

G = 9.81


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _get(ns, name):
    return ns.get(name, "__MISSING__")


def _absent(v):
    return isinstance(v, str) and v == "__MISSING__"


def _num(v):
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except Exception:                                  # noqa: BLE001
        return None


def _close(a, b, tol=1e-4):
    fa, fb = _num(a), _num(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fb))


def _callable_or_reason(ns, name):
    """Return (fn, None) or (None, message) - the standard 'is it a function' check."""
    v = _get(ns, name)
    if _absent(v):
        return None, ("There is no %s in your notebook. The task asks for a "
                      "FUNCTION with exactly that name, defined with "
                      "def %s(...):" % (name, name))
    if callable(v):
        return v, None
    n = _num(v)
    if n is not None:
        return None, ("%s is the number %s, so you calculated one answer "
                      "instead of writing a function. A function takes its "
                      "inputs as arguments and returns the result, so it can "
                      "be re-used on any values: def %s(...): ... return ..."
                      % (name, _num(v), name))
    return None, ("%s is a %s, not a function. Define it with def %s(...):"
                  % (name, type(v).__name__, name))


def _try_call(fn, args):
    """Call a student function. Returns (value, None) or (None, message)."""
    try:
        return fn(*args), None
    except TypeError as exc:
        return None, ("Calling it as %s(%s) raised TypeError: %s. Check how "
                      "many arguments your def line takes, and in which order."
                      % (getattr(fn, "__name__", "f"),
                         ", ".join(repr(a) for a in args), str(exc)[:120]))
    except ZeroDivisionError:
        return None, ("Calling it with %s divided by zero. Guard the case "
                      "before you divide." % (", ".join(repr(a) for a in args),))
    except Exception as exc:                           # noqa: BLE001
        return None, ("Calling it as %s(%s) raised %s: %s"
                      % (getattr(fn, "__name__", "f"),
                         ", ".join(repr(a) for a in args),
                         type(exc).__name__, str(exc)[:120]))


def _no_return(name):
    return ("%s runs but hands back None, so it has no return statement - or "
            "the return is indented inside an if that did not happen. print() "
            "shows a value on screen; return gives it back to the caller, and "
            "the grader can only see what is returned." % name)


# --------------------------------------------------------------------------- #
# q1 - variables and one formula
# --------------------------------------------------------------------------- #

def check_q1(ns):
    v = _get(ns, "weight_n")
    if _absent(v):
        return (False,
                "There is no variable called weight_n. Create it with "
                "weight_n = mass_kg * 9.81, using lower case and an "
                "underscore exactly as written.")
    f = _num(v)
    if f is None:
        if isinstance(v, str):
            return (False,
                    "weight_n is the text %r. Quotation marks make a string, "
                    "and Python will not do arithmetic on it. Write the "
                    "number without quotes." % v[:30])
        return (False, "weight_n is a %s, not a number." % type(v).__name__)
    if _close(f, 40.0):
        return (False,
                "weight_n is 40, which is the MASS in kilograms. Weight is a "
                "force: multiply the mass by g = 9.81 to get newtons.")
    if _close(f, 40.0 / G):
        return (False,
                "weight_n is %.4g, so you divided by 9.81. Weight = mass x g, "
                "so multiply." % f)
    if _close(f, 40.0 * 9.8) and not _close(f, 40.0 * G):
        return (False,
                "weight_n is 392.0, which uses g = 9.8. The task sheet fixes "
                "g at 9.81, so the expected answer is 392.4. Small thing, but "
                "the grader compares numbers.")
    if not _close(f, 40.0 * G):
        return (False,
                "weight_n is %.6g; it should be 392.4 (40 kg x 9.81). Print "
                "your mass and your g separately and check each one." % f)
    return (True, "")


# --------------------------------------------------------------------------- #
# q2 - a function
# --------------------------------------------------------------------------- #

def check_q2(ns):
    fn, why = _callable_or_reason(ns, "percent_error")
    if fn is None:
        return (False, why)
    cases = [((110.0, 100.0), 10.0), ((95.0, 100.0), -5.0),
             ((7.5, 6.0), 25.0), ((6.0, 7.5), -20.0)]
    for args, want in cases:
        got, why = _try_call(fn, args)
        if why:
            return (False, why)
        if got is None:
            return (False, _no_return("percent_error"))
        f = _num(got)
        if f is None:
            if isinstance(got, str):
                return (False,
                        "percent_error(%s, %s) returned the text %r. Return "
                        "the number itself and leave the formatting to "
                        "whoever prints it." % (args[0], args[1], got[:30]))
            return (False, "percent_error returned a %s, not a number."
                    % type(got).__name__)
        if _close(f, want):
            continue
        if _close(f, abs(want)) and want < 0:
            return (False,
                    "percent_error(%s, %s) returned %+.4g but should return "
                    "%+.4g. You have taken the absolute value, which throws "
                    "away whether the measurement was high or low. Keep the "
                    "sign: 100 * (measured - expected) / expected."
                    % (args[0], args[1], f, want))
        if _close(f, 100.0 * (args[0] - args[1]) / args[0]):
            return (False,
                    "percent_error(%s, %s) returned %.4g. You divided by the "
                    "MEASURED value; percent error is measured against the "
                    "expected value, so expected goes on the bottom."
                    % (args[0], args[1], f))
        if _close(f, (args[0] - args[1]) / args[1]):
            return (False,
                    "percent_error(%s, %s) returned %.6g, which is the "
                    "fraction. The task asks for a percentage, so multiply by "
                    "100 - the answer should be %+.4g."
                    % (args[0], args[1], f, want))
        return (False,
                "percent_error(%s, %s) returned %.6g; it should be %+.4g. The "
                "formula is 100 * (measured - expected) / expected."
                % (args[0], args[1], f, want))
    return (True, "")


# --------------------------------------------------------------------------- #
# q3 - the per-cohort function
# --------------------------------------------------------------------------- #

def _check_formula(ns, name, cases, hints, fix_line):
    """Shared body for the three q3 variants.

    cases: list of (args, expected). hints: list of (fn(args) -> expected wrong
    answer, message) tried in order when the value is wrong.
    """
    fn, why = _callable_or_reason(ns, name)
    if fn is None:
        return (False, why)
    for args, want in cases:
        got, why = _try_call(fn, args)
        if why:
            return (False, why)
        if got is None:
            return (False, _no_return(name))
        f = _num(got)
        if f is None:
            return (False, "%s%r returned a %s, not a number."
                    % (name, args, type(got).__name__))
        if _close(f, want):
            continue
        for wrong_fn, message in hints:
            try:
                wrong = wrong_fn(args)
            except Exception:                          # noqa: BLE001
                continue
            if wrong is not None and _close(f, wrong):
                return (False, "%s%r returned %.6g. %s" % (name, args, f, message))
        return (False,
                "%s%r returned %.6g; it should be %.6g. %s"
                % (name, args, f, want, fix_line))
    return (True, "")


def check_q3_cml3111(ns):
    # Resistive loss in a cable: P = I^2 R
    return _check_formula(
        ns, "power_loss",
        [((10.0, 2.5), 250.0), ((3.0, 8.0), 72.0), ((0.5, 100.0), 25.0)],
        [(lambda a: a[0] * a[1],
          "That is I x R, which is a VOLTAGE in volts. Power lost in a "
          "resistance is I squared times R."),
         (lambda a: a[0] * a[1] * a[1],
          "You squared the resistance instead of the current."),
         (lambda a: (a[0] * a[0]) / a[1],
          "You divided by the resistance. I squared R over R would be the "
          "case for a fixed voltage; here the current is given, so multiply.")],
        "Power lost in a resistance is current squared times resistance: "
        "return current ** 2 * resistance.")


def check_q3_cml3112(ns):
    # Mass of a concrete slab: m = area x thickness x 2400 kg/m3
    return _check_formula(
        ns, "slab_mass",
        [((20.0, 0.15), 7200.0), ((100.0, 0.2), 48000.0),
         ((7.5, 0.125), 2250.0)],
        [(lambda a: a[0] * a[1],
          "That is the VOLUME in cubic metres. Multiply the volume by the "
          "density of concrete, 2400 kg per cubic metre, to get mass."),
         (lambda a: a[0] * a[1] * 2400 * 9.81,
          "That is the slab's WEIGHT in newtons. The task asks for mass in "
          "kilograms, so leave g out of it."),
         (lambda a: a[0] * a[1] * 2.4,
          "You used 2.4 rather than 2400. Concrete is about 2400 kg per cubic "
          "metre; 2.4 would be tonnes per cubic metre, which is the same "
          "density in different units.")],
        "Mass = area x thickness x 2400: return area_m2 * thickness_m * 2400.")


def check_q3_cml3113(ns):
    # Hydrostatic pressure in a well: p = rho g h, answer in kPa
    return _check_formula(
        ns, "mud_pressure",
        [((1000.0, 1200.0), 11772.0), ((2500.0, 1050.0), 25753.125),
         ((500.0, 1000.0), 4905.0)],
        [(lambda a: a[0] * a[1] * G,
          "That is the pressure in PASCALS. The task asks for kilopascals, so "
          "divide by 1000. Note that density x g x depth and depth x density "
          "x g give the same number, so the argument order in your def line "
          "is not the problem here."),
         (lambda a: a[0] * a[1] / 1000.0,
          "You left g out. Hydrostatic pressure is density x 9.81 x depth."),
         (lambda a: a[1] * G * a[1] / 1000.0,
          "You used the density twice and never used the depth. Check which "
          "argument is which in your def line: depth comes first.")],
        "Pressure in kPa = density x 9.81 x depth / 1000: "
        "return density * 9.81 * depth_m / 1000.")


# --------------------------------------------------------------------------- #
# the spec
# --------------------------------------------------------------------------- #

_Q3_VARIANTS = {
    "CML3111": {
        "prompt": "Write power_loss(current, resistance) returning the power "
                  "lost in watts (I squared R).",
        "check": check_q3_cml3111,
        "answer": lambda ns: _call_safe(ns, "power_loss", (10.0, 2.5)),
    },
    "CML3112": {
        "prompt": "Write slab_mass(area_m2, thickness_m) returning the mass "
                  "of the concrete slab in kg, taking concrete as 2400 kg/m3.",
        "check": check_q3_cml3112,
        "answer": lambda ns: _call_safe(ns, "slab_mass", (20.0, 0.15)),
    },
    "CML3113": {
        "prompt": "Write mud_pressure(depth_m, density) returning the "
                  "hydrostatic pressure in kPa.",
        "check": check_q3_cml3113,
        "answer": lambda ns: _call_safe(ns, "mud_pressure", (1000.0, 1200.0)),
    },
}
_Q3_VARIANTS["default"] = _Q3_VARIANTS["CML3111"]


def _call_safe(ns, name, args):
    """Fingerprint helper: the return value of a student function, or a marker."""
    fn = ns.get(name)
    if not callable(fn):
        return "__NOT_CALLABLE__"
    try:
        return fn(*args)
    except Exception:                                  # noqa: BLE001
        return "__RAISED__"


QUESTIONS = {
    "q1": {
        "marks": 3, "visible": True,
        "prompt": "A 40 kg motor sits on a bench. Set weight_n to its weight "
                  "in newtons, taking g = 9.81.",
        "check": check_q1,
        "answer_vars": ["weight_n"],
    },
    "q2": {
        "marks": 4, "visible": True,
        "prompt": "Write percent_error(measured, expected) returning the "
                  "signed percent error.",
        "check": check_q2,
        "answer": lambda ns: [_call_safe(ns, "percent_error", a)
                              for a in ((110.0, 100.0), (6.0, 7.5))],
    },
    "q3": {
        "marks": 3, "visible": True,
        "prompt": "Cohort formula - see your task sheet.",
        "check": check_q3_cml3111,
        "variants": _Q3_VARIANTS,
    },
    "q4": {
        "marks": 5, "visible": False,
        "prompt": "Write efficiency(out_w, in_w) returning the percentage, "
                  "and returning None when in_w is zero or negative.",
    },
    "q5": {
        "marks": 5, "visible": False,
        "prompt": "READINGS is given in the notebook. Using your own "
                  "percent_error against a nominal of 100, set worst_reading "
                  "to the reading with the largest error by size, and "
                  "worst_error to that reading's signed percent error.",
    },
}
