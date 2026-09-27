"""iuea_grader - self-check and submission for the IUEA Year 3 Engineering ML course.

One file, Python standard library only, no network. It behaves identically when
imported in Google Colab, in a local Jupyter notebook, in a plain script, and on
an offline lab machine.

Student use, inside a notebook:

    from iuea_grader import Grader
    g = Grader("W05")
    g.check("q2")          # instant feedback on one task
    g.check_all()          # every visible task, plus a progress summary
    g.submit("S24001")     # prints the code to paste into the Google Form

Lecturer use:

    from iuea_grader import verify_token
    verify_token("IUEA1-....-....")

Design notes that matter if you change this file:

* The module reads student variables out of the CALLING frame's globals. In a
  notebook that frame is the user namespace, so `df` defined in cell 4 is
  visible to `g.check("q1")` in cell 5 without the student passing anything.
* `check()` never runs hidden tests and never prints hidden test content. Hidden
  tests live in the same spec file and are run only by batch_grade.py.
* Nothing here may raise. A student must never see a traceback from this module.
  Every public entry point is wrapped by @_safe.
* The HMAC key ships inside this file, which the student can read. The token is
  therefore forgeable by a determined student. See README.md LIMITATIONS. The
  signature stops casual edits and typos, and it lets batch_grade.py tell a
  mangled paste apart from a real score.
"""

import base64
import hashlib
import hmac
import importlib.util
import os
import sys
import time

__version__ = "1.0"

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

# Change this once, at the start of the semester, and keep the same value in
# gradebook.gs (SIGNING_KEY) and in every spec folder you hand out. Changing it
# mid-semester invalidates every token already collected.
SIGNING_KEY = os.environ.get(
    "IUEA_GRADER_KEY", "IUEA-CML31xx-2026S1-change-me"
).encode("utf-8")

TOKEN_PREFIX = "IUEA1"
_HASH_BYTES = 8      # answer fingerprint, 8 bytes -> 13 base32 chars
_SIG_BYTES = 10      # signature, 10 bytes -> 16 base32 chars
_FLOAT_FMT = "%.6g"  # 6 significant figures, so pandas/numpy version drift
                     # does not change the fingerprint

# Where to look for spec_<ASSIGNMENT>.py. Order matters: an explicit path wins,
# then the environment variable, then the usual Colab and local locations.
_SPEC_DIR_ENV = "IUEA_SPEC_DIR"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _safe(fn):
    """Turn any unexpected exception into a printed message and a None return.

    The student sees a sentence telling them what to do next. The lecturer can
    set IUEA_GRADER_DEBUG=1 to get the traceback back.
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                       # noqa: BLE001
            if os.environ.get("IUEA_GRADER_DEBUG"):
                raise
            print("")
            print("  The checker hit a problem it could not handle.")
            print("  %s: %s" % (type(exc).__name__, str(exc)[:300]))
            print("  Nothing is broken in your notebook. Try Runtime > Restart")
            print("  and run all cells from the top. If it happens again, send")
            print("  this message and your notebook to the lecturer.")
            print("")
            return None
    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _b32(raw):
    """base32 without padding: uppercase, no '=', safe to type and to paste."""
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def _unb32(text):
    text = text.strip().upper().replace(" ", "")
    pad = (-len(text)) % 8
    return base64.b32decode(text + "=" * pad)


def _num(x):
    try:
        f = float(x)
    except Exception:                                  # noqa: BLE001
        return "nan"
    if f != f:
        return "nan"
    if f == float("inf"):
        return "inf"
    if f == float("-inf"):
        return "-inf"
    if f == 0.0:
        return "0"
    return _FLOAT_FMT % f


def _canon(value, depth=0):
    """A deterministic short string for any student answer value.

    Used only for the answer fingerprint in the submission token. It must give
    the same result in the student's Colab session and in the lecturer's batch
    run, so everything numeric is rounded to 6 significant figures and pandas
    objects are reduced to shape + column names + per-column digests. pandas and
    numpy are handled by duck typing; this module imports neither.
    """
    if depth > 4:
        return "?"
    if value is None:
        return "N"
    if isinstance(value, bool):
        return "B1" if value else "B0"

    # numpy scalar: has .item() and an empty shape. Must come before int/float
    # because numpy.int64 is not a Python int.
    shape = getattr(value, "shape", None)
    if shape == () and hasattr(value, "item"):
        try:
            return _canon(value.item(), depth + 1)
        except Exception:                              # noqa: BLE001
            pass

    if isinstance(value, int):
        return "I%d" % value
    if isinstance(value, float):
        return "F" + _num(value)
    if isinstance(value, str):
        return "S" + value.strip()
    if isinstance(value, bytes):
        return "Y" + hashlib.sha256(value).hexdigest()[:12]

    # pandas DataFrame
    if hasattr(value, "columns") and shape is not None:
        try:
            cols = [str(c) for c in list(value.columns)]
            parts = ["D%dx%d" % (shape[0], shape[1]), "|".join(cols)]
            for c in cols:
                parts.append(c + "=" + _col_digest(value[c]))
            return ";".join(parts)
        except Exception:                              # noqa: BLE001
            return "D?"

    # pandas Series
    if hasattr(value, "index") and shape is not None and len(shape) == 1:
        try:
            return "R%d;%s" % (shape[0], _col_digest(value))
        except Exception:                              # noqa: BLE001
            return "R?"

    # numpy ndarray
    if shape is not None and hasattr(value, "tolist"):
        try:
            return "A" + "x".join(str(s) for s in shape) + ";" + _canon(
                value.tolist(), depth + 1)
        except Exception:                              # noqa: BLE001
            return "A?"

    if isinstance(value, (list, tuple)):
        inner = ",".join(_canon(v, depth + 1) for v in value[:200])
        return ("L" if isinstance(value, list) else "T") + "%d[%s]" % (
            len(value), inner)
    if isinstance(value, (set, frozenset)):
        return "E%d[%s]" % (
            len(value), ",".join(sorted(_canon(v, depth + 1) for v in value)))
    if isinstance(value, dict):
        items = sorted((str(k), _canon(v, depth + 1)) for k, v in value.items())
        return "M%d[%s]" % (len(items), ",".join(k + ":" + v for k, v in items))

    return "O" + type(value).__name__


def _col_digest(col):
    """Digest one pandas column: sum for numbers, distinct count for anything else."""
    try:
        total = float(col.sum())
        if total == total:                             # not NaN
            return _num(total)
    except Exception:                                  # noqa: BLE001
        pass
    try:
        vals = [str(v) for v in col.tolist()[:500]]
        return "s%d/%d" % (len(set(vals)), len(vals))
    except Exception:                                  # noqa: BLE001
        return "s?"


def _caller_globals():
    """Globals of the nearest frame that is not this module.

    In Colab and in Jupyter the notebook's own namespace IS that frame's
    globals, so student variables are visible without being passed in. In a
    plain script it is the script's module globals. Both are the same mechanism.
    """
    mine = globals()
    frame = sys._getframe(1)                           # noqa: SLF001
    while frame is not None:
        g = frame.f_globals
        if g is not mine and g.get("__name__") != __name__:
            return g
        frame = frame.f_back
    return {}


# --------------------------------------------------------------------------- #
# spec loading
# --------------------------------------------------------------------------- #

def _candidate_dirs():
    here = os.path.dirname(os.path.abspath(__file__ or "."))
    out = []
    env = os.environ.get(_SPEC_DIR_ENV)
    if env:
        out.append(env)
    for base in (os.getcwd(), here, "/content",
                 "/content/drive/MyDrive/IUEA_ML",
                 os.path.expanduser("~/IUEA_ML")):
        out.append(base)
        out.append(os.path.join(base, "specs"))
        out.append(os.path.join(base, "grader", "specs"))
    seen, uniq = set(), []
    for d in out:
        if d and d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def find_spec_file(assignment, spec_path=None):
    """Locate spec_<assignment>.py, or return None."""
    if spec_path:
        return spec_path if os.path.isfile(spec_path) else None
    name = "spec_%s.py" % assignment
    for d in _candidate_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def find_hidden_spec_file(assignment, hidden_path=None):
    """Locate spec_<assignment>_hidden.py, or return None.

    The hidden file is the lecturer's copy and is never handed out. Because
    the student must have the handout spec on their own machine for
    g.check() to work, any check function in the handout can be read by the
    student. So the hidden tests live in a separate file, in a `hidden/`
    subfolder, and only batch_grade.py loads it.
    """
    if hidden_path:
        return hidden_path if os.path.isfile(hidden_path) else None
    name = "spec_%s_hidden.py" % assignment
    for d in _candidate_dirs():
        for sub in ("", "hidden"):
            p = os.path.join(d, sub, name)
            if os.path.isfile(p):
                return p
    return None


def _import_by_path(path, modname):
    mod_spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return module


def load_spec(assignment, spec_path=None, include_hidden=False,
              hidden_path=None):
    """Import a spec file and return the module. Raises on failure.

    include_hidden=True also loads spec_<assignment>_hidden.py, if it exists,
    and merges its QUESTIONS over the handout's. Only batch_grade.py does
    this. The Grader a student uses never does, so a student's copy of the
    handout spec is all they can read, and it has no hidden check functions
    in it.
    """
    path = find_spec_file(assignment, spec_path)
    if path is None:
        raise IOError(
            "cannot find spec_%s.py. Looked in: %s"
            % (assignment, ", ".join(_candidate_dirs()[:6])))
    module = _import_by_path(path, "_iuea_spec_%s" % assignment)
    if not hasattr(module, "QUESTIONS"):
        raise ValueError("%s has no QUESTIONS dictionary" % path)
    module.__spec_path__ = path
    module.__hidden_path__ = None
    if include_hidden:
        hp = find_hidden_spec_file(assignment, hidden_path)
        if hp:
            hidden = _import_by_path(hp, "_iuea_hidden_%s" % assignment)
            extra = getattr(hidden, "QUESTIONS", {}) or {}
            merged = dict(module.QUESTIONS)
            for qid, q in extra.items():
                base = dict(merged.get(qid, {}))
                base.update(q)
                base["visible"] = False
                merged[qid] = base
            module.QUESTIONS = merged
            module.__hidden_path__ = hp
            order = list(getattr(module, "QUESTION_ORDER", None) or
                         module.QUESTIONS.keys())
            for qid in extra:
                if qid not in order:
                    order.append(qid)
            module.QUESTION_ORDER = order
    return module


def resolve_questions(spec, cohort=None):
    """Apply per-cohort variants and return an ordered list of (qid, question).

    A question may carry a "variants" dict keyed by course code. The matching
    variant's keys replace the base question's keys for that cohort. Order is
    QUESTION_ORDER if the spec defines it, otherwise the dictionary order.
    """
    order = getattr(spec, "QUESTION_ORDER", None) or list(spec.QUESTIONS.keys())
    out = []
    for qid in order:
        q = dict(spec.QUESTIONS[qid])
        variants = q.pop("variants", None) or {}
        if cohort and cohort in variants:
            q.update(variants[cohort])
        elif variants and "default" in variants:
            q.update(variants["default"])
        q.setdefault("marks", 1)
        q.setdefault("visible", True)
        q.setdefault("prompt", qid)
        out.append((qid, q))
    return out


# --------------------------------------------------------------------------- #
# running one check
# --------------------------------------------------------------------------- #

class Result(object):
    """Outcome of one question."""

    __slots__ = ("qid", "ok", "awarded", "marks", "message", "error")

    def __init__(self, qid, ok, awarded, marks, message="", error=""):
        self.qid = qid
        self.ok = ok
        self.awarded = awarded
        self.marks = marks
        self.message = message
        self.error = error

    def __repr__(self):
        return "<Result %s %s %.4g/%.4g>" % (
            self.qid, "PASS" if self.ok else "FAIL", self.awarded, self.marks)


def run_check(qid, question, namespace):
    """Run one question's check function against a namespace. Never raises."""
    marks = float(question.get("marks", 1))
    check = question.get("check")
    if not callable(check):
        return Result(qid, False, 0.0, marks,
                      "This question has no check function. Tell the lecturer.",
                      error="BAD_SPEC")
    try:
        verdict = check(namespace)
    except Exception as exc:                           # noqa: BLE001
        # A check that explodes is a fail for the student and a bug report for
        # the lecturer. It is never a crash.
        return Result(qid, False, 0.0, marks,
                      "The check for this task could not run: %s: %s"
                      % (type(exc).__name__, str(exc)[:200]),
                      error="CHECK_RAISED")
    fraction = None
    if isinstance(verdict, tuple):
        ok = bool(verdict[0])
        message = str(verdict[1]) if len(verdict) > 1 else ""
        if len(verdict) > 2:
            try:
                fraction = float(verdict[2])
            except Exception:                          # noqa: BLE001
                fraction = None
    else:
        ok, message = bool(verdict), ""
    if fraction is None:
        fraction = 1.0 if ok else 0.0
    fraction = max(0.0, min(1.0, fraction))
    return Result(qid, ok, round(marks * fraction, 4), marks, message)


def answer_fingerprint(qid_question_pairs, namespace):
    """Hash of the student's answer VALUES, over the VISIBLE questions only.

    Visible only, deliberately. The student's Grader has loaded the handout
    spec, which has no hidden check functions in it, so a fingerprint over
    every question would not match the one batch_grade.py computes from the
    merged spec. Filtering to the visible questions makes the two agree
    whichever spec was loaded.


    Recomputed by batch_grade.py from the submitted notebook and compared with
    the value inside the token. A mismatch means the notebook the lecturer
    received does not produce the answers the token claims. Treat it as a flag
    to look at, not as proof of anything: a genuine mismatch also happens if the
    student edits the notebook after submitting, or re-runs it with a different
    pandas version.
    """
    parts = []
    for qid, q in qid_question_pairs:
        if not q.get("visible", True):
            continue
        fn = q.get("answer")
        if callable(fn):
            try:
                value = fn(namespace)
            except Exception:                          # noqa: BLE001
                value = "__ERROR__"
        else:
            names = q.get("answer_vars") or []
            value = [namespace.get(n, "__MISSING__") for n in names]
        parts.append(qid + "=" + _canon(value))
    blob = "\n".join(parts).encode("utf-8", "replace")
    return _b32(hashlib.sha256(blob).digest()[:_HASH_BYTES])


# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #

def _fmt_marks(x):
    x = round(float(x), 4)
    return ("%d" % x) if abs(x - int(x)) < 1e-9 else ("%g" % x)


def make_token(student_id, assignment, cohort, visible_scores, fingerprint,
               when=None, key=None):
    """Build the submission code.

    Layout: IUEA1-<base32 of body>-<base32 of truncated HMAC-SHA256 of body>
    body:   sid|assignment|cohort|q1=3,q2=0,q3=4|<fingerprint>|<base36 unix seconds>
    """
    key = key or SIGNING_KEY
    when = int(when if when is not None else time.time())
    scores = ",".join("%s=%s" % (qid, _fmt_marks(v)) for qid, v in visible_scores)
    body = "|".join([
        str(student_id).strip().upper(),
        str(assignment).strip().upper(),
        (str(cohort).strip().upper() if cohort else "-"),
        scores,
        fingerprint,
        _base36(when),
    ])
    raw = body.encode("utf-8")
    sig = hmac.new(key, raw, hashlib.sha256).digest()[:_SIG_BYTES]
    return "%s-%s-%s" % (TOKEN_PREFIX, _b32(raw), _b32(sig))


def _base36(n):
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = int(n)
    if n == 0:
        return "0"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out


def _unbase36(s):
    return int(s, 36)


def verify_token(token, key=None):
    """Check a submission code and unpack it.

    Returns a dict. `ok` is True only if the signature matches. `reason` says
    why not. Never raises.
    """
    out = {"ok": False, "reason": "", "token": (token or "").strip(),
           "student_id": "", "assignment": "", "cohort": "",
           "scores": {}, "claimed_total": 0.0, "fingerprint": "",
           "timestamp": 0, "time_utc": ""}
    key = key or SIGNING_KEY
    text = (token or "").strip().replace(" ", "")
    if not text:
        out["reason"] = "empty"
        return out
    parts = text.split("-")
    if len(parts) != 3 or parts[0].upper() != TOKEN_PREFIX:
        out["reason"] = "not an IUEA submission code"
        return out
    try:
        raw = _unb32(parts[1])
        sig = _unb32(parts[2])
    except Exception:                                  # noqa: BLE001
        out["reason"] = "code is corrupted (bad base32) - probably a bad paste"
        return out
    expect = hmac.new(key, raw, hashlib.sha256).digest()[:_SIG_BYTES]
    if not hmac.compare_digest(sig, expect):
        out["reason"] = "signature does not match"
        try:
            out["body_preview"] = raw.decode("utf-8", "replace")[:120]
        except Exception:                              # noqa: BLE001
            pass
        return out
    try:
        body = raw.decode("utf-8")
        sid, aid, cohort, scores, fp, ts = body.split("|")
    except Exception:                                  # noqa: BLE001
        out["reason"] = "signature valid but body unreadable"
        return out
    parsed, total = {}, 0.0
    for item in scores.split(","):
        if not item:
            continue
        if "=" in item:
            qid, val = item.split("=", 1)
        else:
            qid, val = item, "0"
        try:
            v = float(val)
        except Exception:                              # noqa: BLE001
            v = 0.0
        parsed[qid] = v
        total += v
    try:
        when = _unbase36(ts)
    except Exception:                                  # noqa: BLE001
        when = 0
    out.update({
        "ok": True, "reason": "ok", "student_id": sid, "assignment": aid,
        "cohort": ("" if cohort == "-" else cohort), "scores": parsed,
        "claimed_total": round(total, 4), "fingerprint": fp,
        "timestamp": when,
        "time_utc": time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.gmtime(when)) if when else "",
    })
    return out


# --------------------------------------------------------------------------- #
# the student-facing object
# --------------------------------------------------------------------------- #

_RULE = "-" * 64


class Grader(object):
    """Self-check and submission for one assignment.

        g = Grader("W05")
        g.check("q2")
        g.check_all()
        g.submit("S24001")

    cohort: pass "CML3111" / "CML3112" / "CML3113" to pick a per-cohort variant.
    If omitted, the grader reads a COHORT variable from your notebook, and
    falls back to the spec's DEFAULT_COHORT.
    """

    def __init__(self, assignment, cohort=None, spec_path=None, namespace=None,
                 quiet=False):
        self.assignment = str(assignment).strip().upper()
        self._namespace = namespace
        self._ns_at_init = None if namespace is not None else _caller_globals()
        self.spec = None
        self.questions = []
        self.cohort = ""
        self.title = ""
        self.ok = False
        try:
            self.spec = load_spec(self.assignment, spec_path)
        except Exception as exc:                       # noqa: BLE001
            print(_RULE)
            print("  Could not load the checks for %s." % self.assignment)
            print("  %s" % str(exc)[:400])
            print("")
            print("  Fix: make sure spec_%s.py sits in the same folder as this"
                  % self.assignment)
            print("  notebook. On Colab, run the setup cell at the top again.")
            print(_RULE)
            return
        chosen = cohort or self._detect_cohort() or getattr(
            self.spec, "DEFAULT_COHORT", "") or ""
        self.cohort = str(chosen).strip().upper()
        self.title = getattr(self.spec, "TITLE", self.assignment)
        self.questions = resolve_questions(self.spec, self.cohort)
        self.ok = True
        if not quiet:
            self._banner()

    # -- internals ---------------------------------------------------------- #

    def _ns(self):
        if self._namespace is not None:
            return self._namespace
        # Resolved at call time so variables created after Grader(...) are seen.
        # Two frames up: caller of _ns -> public method -> _safe wrapper -> user.
        mine = globals()
        frame = sys._getframe(1)                       # noqa: SLF001
        while frame is not None:
            g = frame.f_globals
            if g is not mine and g.get("__name__") != __name__:
                return g
            frame = frame.f_back
        return self._ns_at_init or {}

    def _detect_cohort(self):
        ns = self._ns_at_init or {}
        for name in ("COHORT", "cohort", "COURSE_CODE"):
            v = ns.get(name)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _visible(self):
        return [(qid, q) for qid, q in self.questions if q.get("visible", True)]

    def _hidden(self):
        return [(qid, q) for qid, q in self.questions
                if not q.get("visible", True)]

    def _banner(self):
        vis = self._visible()
        hid = self._hidden()
        vis_marks = sum(float(q.get("marks", 1)) for _, q in vis)
        hid_marks = sum(float(q.get("marks", 1)) for _, q in hid)
        print(_RULE)
        print("  %s  %s" % (self.assignment, self.title))
        if self.cohort:
            print("  Cohort: %s" % self.cohort)
        print("  %d tasks you can check yourself, worth %s marks."
              % (len(vis), _fmt_marks(vis_marks)))
        if hid:
            print("  A further %s marks are graded after you submit."
                  % _fmt_marks(hid_marks))
        print("  Check one task:  g.check(\"%s\")" % (vis[0][0] if vis else "q1"))
        print("  Check them all:  g.check_all()")
        print("  Submit:          g.submit(\"YOUR_STUDENT_ID\")")
        print(_RULE)

    def _find(self, qid):
        want = str(qid).strip().lower()
        for q_id, q in self.questions:
            if q_id.lower() == want:
                return q_id, q
        return None, None

    # -- output formatting -------------------------------------------------- #

    @staticmethod
    def _print_result(res, prompt):
        head = "  [PASS]" if res.ok else "  [FAIL]"
        if 0 < res.awarded < res.marks:
            head = "  [PART]"
        print("%s %s   %s / %s marks" % (
            head, res.qid, _fmt_marks(res.awarded), _fmt_marks(res.marks)))
        if not res.ok or res.awarded < res.marks:
            if prompt:
                print("         Task: %s" % prompt)
            for line in _wrap(res.message or
                              "Something is not right yet. Re-read the task.",
                              72):
                print("         %s" % line)
        print("")

    # -- public API --------------------------------------------------------- #

    @_safe
    def check(self, qid):
        """Check ONE visible task and print feedback. Returns the Result."""
        if not self.ok:
            print("  The checks are not loaded, so there is nothing to check.")
            return None
        q_id, q = self._find(qid)
        if q is None:
            names = ", ".join(i for i, _ in self._visible())
            print("  There is no task called %r in %s." % (qid, self.assignment))
            print("  The ones you can check are: %s" % names)
            return None
        if not q.get("visible", True):
            print("  %s is graded by the lecturer after you submit, so there is"
                  % q_id)
            print("  no self-check for it. Do the task as the sheet describes,")
            print("  using the exact variable names it asks for.")
            return None
        res = run_check(q_id, q, self._ns())
        self._print_result(res, q.get("prompt", ""))
        return res

    @_safe
    def check_all(self):
        """Check every visible task and print a progress summary."""
        if not self.ok:
            print("  The checks are not loaded, so there is nothing to check.")
            return []
        vis = self._visible()
        ns = self._ns()
        results = [run_check(qid, q, ns) for qid, q in vis]
        print(_RULE)
        print("  %s  self-check" % self.assignment)
        print(_RULE)
        for res, (_, q) in zip(results, vis):
            self._print_result(res, q.get("prompt", ""))
        passed = sum(1 for r in results if r.ok)
        got = sum(r.awarded for r in results)
        avail = sum(r.marks for r in results)
        print(_RULE)
        print("  You have passed %d of %d checks: %s of %s visible marks."
              % (passed, len(results), _fmt_marks(got), _fmt_marks(avail)))
        hid = self._hidden()
        if hid:
            hm = sum(float(q.get("marks", 1)) for _, q in hid)
            print("  %s further marks are graded from your notebook after you"
                  % _fmt_marks(hm))
            print("  submit. Passing every check here does not guarantee those.")
        if passed == len(results) and results:
            print("  All checks pass. Run g.submit(\"YOUR_STUDENT_ID\").")
        else:
            todo = ", ".join(r.qid for r in results if not r.ok)
            print("  Still to fix: %s" % todo)
        print(_RULE)
        return results

    @_safe
    def submit(self, student_id):
        """Print the submission code to paste into the Google Form."""
        if not self.ok:
            print("  The checks are not loaded, so a code cannot be made.")
            return None
        sid = str(student_id or "").strip().upper()
        if len(sid) < 3 or sid in ("YOUR_STUDENT_ID", "STUDENT_ID"):
            print("  Put your own student number in, like this:")
            print("      g.submit(\"S24001\")")
            return None
        ns = self._ns()
        vis = self._visible()
        results = [run_check(qid, q, ns) for qid, q in vis]
        fp = answer_fingerprint(self.questions, ns)
        token = make_token(sid, self.assignment, self.cohort,
                           [(r.qid, r.awarded) for r in results], fp)
        got = sum(r.awarded for r in results)
        avail = sum(r.marks for r in results)
        failed = [r.qid for r in results if not r.ok]
        print(_RULE)
        print("  SUBMISSION CODE for %s, %s" % (sid, self.assignment))
        print(_RULE)
        print("")
        print("  %s" % token)
        print("")
        print("  (%d characters. Copy the whole line, including IUEA1.)" % len(token))
        print(_RULE)
        print("  Visible marks in this code: %s of %s."
              % (_fmt_marks(got), _fmt_marks(avail)))
        if failed:
            print("  Not passing yet: %s. You can fix them and submit again;"
                  % ", ".join(failed))
            print("  the lecturer counts your last code before the deadline.")
        print("  Next: paste the code into the Google Form for %s, then upload"
              % self.assignment)
        print("  this notebook to the same form. Both are required - the code")
        print("  on its own is not a submission.")
        print(_RULE)
        return token

    @_safe
    def grade_all(self, namespace=None):
        """Lecturer only: run visible AND hidden checks. Returns Results."""
        ns = namespace if namespace is not None else self._ns()
        return [run_check(qid, q, ns) for qid, q in self.questions]

    @_safe
    def tasks(self):
        """Print the task list, marks, and which ones you can self-check."""
        if not self.ok:
            return None
        print(_RULE)
        for qid, q in self.questions:
            tag = "self-check" if q.get("visible", True) else "graded later"
            print("  %-4s %-13s %s marks" % (
                qid, tag, _fmt_marks(q.get("marks", 1))))
            if q.get("visible", True):
                for line in _wrap(q.get("prompt", ""), 66):
                    print("         %s" % line)
        print(_RULE)
        return None


def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


# --------------------------------------------------------------------------- #
# command line: a self-test that needs no notebook and no network
# --------------------------------------------------------------------------- #

def _selftest():
    ok = fail = 0

    def t(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print("  pass  %s" % name)
        else:
            fail += 1
            print("  FAIL  %s" % name)

    tok = make_token("S24001", "W05", "CML3111",
                     [("q1", 3), ("q2", 0), ("q3", 4)], "ABCDEFGHIJKLM",
                     when=1757000000)
    t("token under 200 chars (%d)" % len(tok), len(tok) < 200)
    v = verify_token(tok)
    t("round trip verifies", v["ok"])
    t("student id survives", v["student_id"] == "S24001")
    t("scores survive", v["scores"] == {"q1": 3.0, "q2": 0.0, "q3": 4.0})
    t("claimed total", v["claimed_total"] == 7.0)
    t("fingerprint survives", v["fingerprint"] == "ABCDEFGHIJKLM")
    t("timestamp survives", v["timestamp"] == 1757000000)

    bad = tok[:-3] + ("AAA" if not tok.endswith("AAA") else "BBB")
    t("tampered signature rejected", not verify_token(bad)["ok"])
    forged = make_token("S24001", "W05", "CML3111",
                        [("q1", 3), ("q2", 3), ("q3", 4)], "ABCDEFGHIJKLM",
                        when=1757000000, key=b"wrong-key")
    t("token signed with wrong key rejected", not verify_token(forged)["ok"])
    t("empty token rejected", not verify_token("")["ok"])
    t("junk token rejected", not verify_token("hello world")["ok"])
    t("None token rejected", not verify_token(None)["ok"])

    t("canon int stable", _canon(5) == _canon(5))
    t("canon float rounds", _canon(1.0000000001) == _canon(1.0))
    t("canon distinguishes", _canon(5) != _canon(6))
    t("canon handles None", _canon(None) == "N")
    t("canon handles bool before int", _canon(True) == "B1")
    t("canon deep recursion terminates", isinstance(_canon([[[[[[1]]]]]]), str))

    print("")
    print("  %d passed, %d failed" % (ok, fail))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--selftest", "--self-test"):
        sys.exit(_selftest())
    if len(sys.argv) > 2 and sys.argv[1] == "--verify":
        info = verify_token(sys.argv[2])
        for k in ("ok", "reason", "student_id", "assignment", "cohort",
                  "scores", "claimed_total", "fingerprint", "time_utc"):
            print("%-14s %s" % (k, info.get(k)))
        sys.exit(0 if info["ok"] else 1)
    print(__doc__)
    print("Try:  python3 iuea_grader.py --selftest")
    print("      python3 iuea_grader.py --verify IUEA1-...-...")
