"""Shared opt-in guard for superseded Rayleigh--Bénard analysis paths."""


def guard_legacy_rb(dataset: str | None, allow_legacy: bool) -> None:
    if dataset == "rayleigh_benard" and not allow_legacy:
        raise SystemExit(
            "legacy RB analysis is disabled: its derivatives, quadrature, target suite, "
            "and 101-frame sampling are invalid for physical claims. Use scripts/rb_eval_v2.py, "
            "or pass --allow-legacy-rb only for a documented forensic reproduction."
        )
